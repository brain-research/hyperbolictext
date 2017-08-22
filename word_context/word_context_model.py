#!/usr/bin/python
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Version of Skip Thoughts which decodes a fixed window of context words."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from skip_thoughts import skip_thoughts_model
from skip_thoughts.ops import input_ops
from word_context import metrics


EPSILON = 1e-15
INFINITY = 1e15


def _parse_single_example(serialized):
  """Parses a single tf.Example proto.

  Args:
    serialized: A 0D string Tensor; the serialized tf.Example proto.
  Returns:
    encode: A 1D int64 Tensor; the sentence to encode.
    decode_pre: A 1D int64 Tensor; the "previous" sentence to decode.
    decode_post: A 1D int64 Tensor; the "post" sentence to decode.
  """
  features = tf.parse_single_example(
      serialized,
      features={
          "encode": tf.VarLenFeature(dtype=tf.int64),
          "decode_pre": tf.VarLenFeature(dtype=tf.int64),
          "decode_post": tf.VarLenFeature(dtype=tf.int64),
      })

  encode = features["encode"].values
  decode_pre = features["decode_pre"].values
  decode_post = features["decode_post"].values

  return encode, decode_pre, decode_post


def _pad_to_min_length(seq, min_len, pad_from_front=False):
  """Ensures the minimum length of a sequence, padding with zeros if necessary.

  Args:
    seq: A 1-D tensor.
    min_len: Minimum desired length of sequence.
    pad_from_front: If True, pad from the front of the sequence. Otherwise pad
      from the end.

  Returns:
    seq_out: Padded sequence of shape [length]; 1-D tensor.
    mask: 1D Tensor indicating which elements are real.
  """
  seq_len = tf.size(seq)
  len_diff = tf.subtract(min_len, seq_len)
  mask = tf.ones_like(seq, dtype=tf.int32)

  def pad():
    pad_spec = [len_diff, 0] if pad_from_front else [0, len_diff]
    pad_arg = tf.expand_dims(tf.stack(pad_spec), 0)
    return tf.pad(seq, pad_arg), tf.pad(mask, pad_arg)

  def identity():
    return seq, mask

  seq_out, mask = tf.cond(tf.greater(len_diff, 0), pad, identity)
  return seq_out, mask


def _custom_norm(tensor):
  """Stable norm implementation.

  Args:
    tensor: 2D tensor.

  Returns:
    norms: 1D tensor.
  """
  return tf.sqrt(
      tf.maximum(EPSILON, tf.reduce_sum(tf.square(tensor), axis=1)))


def _project_to_unit_sphere(tensor):
  """Project tensor rows with norm > 1 to the unit sphere.

  Args:
    tensor: 2D Tensor.

  Returns:
    projected_tensor: 2D Tensor whose rows have norm < 1.
  """
  EPS = 1e-3
  row_norms = _custom_norm(tensor)
  overflow = tf.greater_equal(row_norms, 1.-EPS)
  scaled_embeddings = tensor / tf.expand_dims(row_norms + EPS, axis=1)
  projected_tensor = tf.where(overflow, scaled_embeddings, tensor)
  return projected_tensor


def _independent_norm_scaling(tensor, norm_fn, shift):
  """Reparameterize tensor by setting the norm to last dimension.

  The last dimension is is projected to [0, 1) using norm_fn.

  Args:
    tensor: 2D Tensor.
    norm_fn: A element-wise function which returns values in [0,1).
    shift: Constant shift before passing to norm_fn.

  Returns:
    reparameterized_tensor: 2D Tensor with shape[1] = tensor.shape[1] - 1,
      and whose rows have norm < 1.
  """
  scaled_norms = norm_fn(tensor[:,-1] + shift)
  reparameterized_tensor = tf.nn.l2_normalize(tensor[:,:-1], dim=1)
  reparameterized_tensor *= tf.expand_dims(scaled_norms, axis=1)
  return reparameterized_tensor, scaled_norms


def _gaussian_norm_scaling(tensor, temperature):
  """Reparameterize tensor by scaling row-norms with gaussian.

  Row-norms of returned tensor are scaled between [0,1).

  Args:
    tensor: 2D Tensor.
    temperature: Constant multiplier for the exponent. Smaller values will lead
      to smaller output norms.

  Returns:
    reparameterized_tensor: 2D Tensor whose rows have norm < 1.
  """
  scaling_factors = 1. - tf.exp( - temperature *
                                tf.reduce_sum(tf.square(tensor), axis=1))

  reparameterized_tensor = tf.nn.l2_normalize(tensor, dim=1) * tf.expand_dims(
      scaling_factors, axis=1)

  return reparameterized_tensor, scaling_factors


class WordContextModel(skip_thoughts_model.SkipThoughtsModel):
  """Version of Skip Thoughts which decodes a fixed window of surrounding words.
  """

  def __init__(self, config, mode="train", input_reader=None):
    """Basic setup. The actual TensorFlow graph is constructed in build().

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "encode".
      input_reader: Subclass of tf.ReaderBase for reading the input serialized
        tf.Example protocol buffers. Defaults to TFRecordReader.

    Raises:
      ValueError: If mode is invalid.
    """
    super(WordContextModel, self).__init__(config, mode, input_reader)

  def build_inputs(self):
    """Builds the ops for reading input data.

    Outputs:
      self.encode_ids
      self.decode_pre_ids
      self.decode_post_ids
      self.encode_mask
      self.decode_pre_mask
      self.decode_post_mask
    """
    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has possibly
      # been expanded (see vocabulary_expansion.py).
      encode_ids = None
      decode_pre_ids = None
      decode_post_ids = None
      encode_mask = tf.placeholder(tf.int32, (None, None), name="encode_mask")
      decode_pre_mask = None
      decode_post_mask = None
    elif self.mode == "decode":
      # Word embeddings are fed from an external vocabulary which has possibly
      # been expanded (see vocabulary_expansion.py).
      encode_ids = tf.placeholder(tf.int64, (None, None), name="encode_ids")
      decode_pre_ids = tf.placeholder(tf.int64, (None, None), name="decode_pre_ids")
      decode_post_ids = tf.placeholder(tf.int64, (None, None), name="decode_post_ids")
      encode_mask = tf.placeholder(tf.int32, (None, None), name="encode_mask")
      decode_pre_mask = tf.placeholder(tf.int32, (None, None),
                                       name="decode_pre_mask")
      decode_post_mask = tf.placeholder(tf.int32, (None, None),
                                        name="decode_post_mask")
    else:
      # Prefetch serialized tf.Example protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          shuffle=self.config.shuffle_input_data,
          capacity=self.config.input_queue_capacity,
          num_reader_threads=self.config.num_input_reader_threads)

      serialized = input_queue.dequeue()
      encode, decode_pre, decode_post = _parse_single_example(serialized)
      encode_mask = tf.ones_like(encode, dtype=tf.int32)

      # Ensure the minimum lengths of decode_pre and decode_post. We request an
      # extra length unit for decode_pre, because we will clip out the <EOS>
      # later (only if decode_strategy != "conditional").
      if self.config.decode_strategy == "conditional":
        decode_pre, decode_pre_mask = _pad_to_min_length(
            decode_pre,
            self.config.context_window + self.config.condition_length,
            pad_from_front=True)
        decode_post, decode_post_mask = _pad_to_min_length(
            decode_post,
            self.config.context_window + self.config.condition_length,
            pad_from_front=False)
      else:
        decode_pre, decode_pre_mask = _pad_to_min_length(
            decode_pre, self.config.context_window + 1, pad_from_front=True)
        decode_post, decode_post_mask = _pad_to_min_length(
            decode_post, self.config.context_window, pad_from_front=False)

      # Clip to the end of decode_pre and the beginning of decode_post. Also
      # ignore the <EOS> at the end of decode_pre (only if decode_strategy !=
      # "conditional").
      if self.config.decode_strategy == "conditional":
        decode_pre = decode_pre[
            -(self.config.context_window + self.config.condition_length):]
        decode_pre_mask = decode_pre_mask[
            -(self.config.context_window + self.config.condition_length):]
        decode_post = decode_post[
            0:self.config.context_window + self.config.condition_length]
        decode_post_mask = decode_post_mask[
            0:self.config.context_window + self.config.condition_length]
      else:
        decode_pre = tf.reverse(
            decode_pre, [0])[1:self.config.context_window + 1]
        decode_pre_mask = tf.reverse(decode_pre_mask,
                                     [0])[1:self.config.context_window + 1]
        decode_post = decode_post[0:self.config.context_window]
        decode_post_mask = decode_post_mask[0:self.config.context_window]

      (encode_ids, decode_pre_ids, decode_post_ids, encode_mask,
       decode_pre_mask, decode_post_mask) = tf.train.batch(
           tensors=[
               encode,
               decode_pre,
               decode_post,
               encode_mask,
               decode_pre_mask,
               decode_post_mask,
           ],
           batch_size=self.config.batch_size,
           num_threads=self.config.num_batching_threads,
           capacity=5 * self.config.batch_size,
           dynamic_pad=True)

    self.encode_ids = encode_ids
    self.decode_pre_ids = decode_pre_ids
    self.decode_post_ids = decode_post_ids

    self.encode_mask = encode_mask
    self.decode_pre_mask = decode_pre_mask
    self.decode_post_mask = decode_post_mask

  def build_encoder(self):
    """Builds the sentence encoder.

    Inputs:
      self.encode_emb
      self.encode_mask

    Outputs:
      self.thought_vectors

    Raises:
      ValueError: if config.bidirectional_encoder is True and config.encoder_dim
        is odd.
    """
    with tf.variable_scope("encoder") as scope:
      length = tf.to_int32(tf.reduce_sum(self.encode_mask, 1), name="length")

      if self.config.bidirectional_encoder:
        if self.config.encoder_dim % 2:
          raise ValueError(
              "encoder_dim must be even when using a bidirectional encoder.")
        num_units = self.config.encoder_dim // 2
        cell_fw = self._initialize_gru_cell(num_units)  # Forward encoder
        cell_bw = self._initialize_gru_cell(num_units)  # Backward encoder
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=self.encode_emb,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope)
        outputs = tf.concat(outputs, 2, name="encoder_outputs")
        state = tf.concat(states, 1)
      else:
        cell = self._initialize_gru_cell(self.config.encoder_dim)
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.encode_emb,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope)
        outputs = tf.identity(outputs, name="encoder_outputs")

      # Pool output vectors into a single fixed-size representation
      if self.config.pooling_operation == "last":
        # Use an identity operation to name the Tensor in the Graph.
        thought_vectors = tf.identity(state, name="thought_vectors_pre")
      elif self.config.pooling_operation == "max":
        # Set padded elements to large negative value
        outputs_before_max = tf.where(
            tf.tile(
                tf.expand_dims(tf.equal(self.encode_mask, 0), axis=2),
                [1, 1, tf.shape(outputs)[2]]),
            -1e5 * tf.ones_like(outputs),
            outputs)
        thought_vectors = tf.reduce_max(
            outputs_before_max, axis=1, name="thought_vectors_pre")
      else:
        raise ValueError("pooling_operation %s unrecognized" %
                         self.config.pooling_operation)

      self.thought_vectors_pre = thought_vectors
      self.encoder_outputs = outputs

      if self.config.decode_strategy == "conditional":
        # Denote B = batch_size, V = vocabulary_size
        if (self.config.logit_metric == "hyperbolic" and
            self.config.reparameterization == "independent"):
          num_units = self.config.encoder_dim + 1
        else:
          num_units = self.config.encoder_dim

        if self.config.logit_metric is None:
          thoughts_rescaled = self.thought_vectors_pre

        elif self.config.logit_metric == "cosine":
          thoughts_rescaled = tf.nn.l2_normalize(self.thought_vectors_pre, 1)

        elif self.config.logit_metric == "hyperbolic":
          # Reparameterize so that norm is always less than 1
          if self.config.reparameterization == "tanh":
            scaling_factor = 1./tf.sqrt(float(self.config.encoder_dim))
            thoughts_rescaled = scaling_factor * self.thought_vectors_pre
          elif self.config.reparameterization == "gaussian":
            thought_initializer = tf.random_uniform_initializer(
                minval=-1./self.config.encoder_dim,
                maxval=1./self.config.encoder_dim)
            thoughts_projected = tf.contrib.layers.fully_connected(
                inputs=self.thought_vectors_pre,
                num_outputs=self.config.encoder_dim,
                activation_fn=None,
                biases_initializer=None,
                weights_initializer=thought_initializer,
                scope="thoughts_projection")
            thought_temperature = 0.9
            thoughts_rescaled, thoughts_scales = _gaussian_norm_scaling(
                thoughts_projected, thought_temperature)
          elif self.config.reparameterization == "independent":
            thought_initializer = tf.random_uniform_initializer(
                minval=-1./self.config.encoder_dim,
                maxval=1./self.config.encoder_dim)
            thoughts_projected = tf.contrib.layers.fully_connected(
                inputs=self.thought_vectors_pre,
                num_outputs=self.config.encoder_dim + 1,
                activation_fn=None,
                biases_initializer=None,
                weights_initializer=thought_initializer,
                scope="thoughts_projection")
            thoughts_rescaled, thoughts_scales = _independent_norm_scaling(
                thoughts_projected, tf.sigmoid,
                self.config.independent_norm_shift)
          elif self.config.reparameterization == "projection":
            thoughts_rescaled = _project_to_unit_sphere(
                self.thought_vectors_pre)
          else:
            raise ValueError("Unrecognized reparameterization: %s" %
                             self.config.reparameterization)

      elif self.config.decode_strategy == "positional":
        thoughts_rescaled = self.thought_vectors_pre

      elif self.config.decode_strategy == "biased":
        thoughts_rescaled = self.thought_vectors_pre
      else:
        raise ValueError("Unrecognized decode_strategy: %s" %
                         self.config.decode_strategy)

      self.thought_vectors = tf.identity(thoughts_rescaled,
                                         name="thought_vectors")

  def build_decoders(self):
    """Builds the sentence decoders.

    Inputs:
      self.decode_pre_emb
      self.decode_post_emb
      self.decode_pre_ids
      self.decode_post_ids
      self.decode_pre_mask
      self.decode_post_mask
      self.thought_vectors

    Outputs:
      self.target_cross_entropy_losses
      self.target_cross_entropy_loss_weights

    Raises:
      ValueError: If config.decode_strategy is invalid.
    """
    if self.mode == "encode":
      return

    # targets is 1D with shape [batch_size * 2 * context_window]:
    # [
    #   batch index 0: decode_pre word 0
    #   batch index 0: decode_pre word 1
    #   ...
    #   batch index 0: decode_post word 0
    #   batch index 0: decode_post word 1
    #   ...
    #   batch index 1: decode_pre word 0
    #   ...
    # ]
    if self.config.decode_strategy == "conditional":
      targets = tf.reshape(
          tf.concat([self.decode_pre_ids[:,self.config.condition_length:],
                     self.decode_post_ids[:,:self.config.context_window]],
                    axis=1),
          [-1])
      target_weights = tf.to_float(
          tf.reshape(
              tf.concat([self.decode_pre_mask[:,self.config.condition_length:],
                         self.decode_post_mask[:,:self.config.context_window]],
                        axis=1),
              [-1]))
    else:
      targets = tf.reshape(
          tf.concat([self.decode_pre_ids, self.decode_post_ids], axis=1), [-1])
      target_weights = tf.to_float(
          tf.reshape(
              tf.concat([self.decode_pre_mask, self.decode_post_mask], axis=1),
              [-1]))

    # Decoder cases:
    #   1) decode_strategy = "conditional": A local context vector is computed
    #      for each each position to be decoded.
    #
    #      There are two options to specify the context:
    #       a) condition_uni_context == True: The context vector is formed by
    #          averaging word vectors in a window of size condition_length
    #          preceding the target word.
    #
    #       b) condition_uni_context == False: The context vector is formed by
    #          averaging word vectors in two windows of size condition_length,
    #          one preceding the target word and one following it.
    #
    #      In addition it is possible to provide the metric for computing
    #      logits:
    #       a) logit_metric == "cosine": Logits are computed as cosine
    #          similarity between output embedding of the target word and the
    #          context vector / thought vector, plus a bias.
    #       b) logit_metric == "hyperbolic": Logits are computed as the
    #          negative hyperbolic distance between the output embedding of the
    #          target word and context vector / thought vector, plus a bias.
    #       c) logit_metric is None: Logits are computed by standard softmax
    #          after adding the thought vector and context vector.
    #
    #   2) decode_strategy = "positional": The encoded sentence is multiplied by
    #      a separate weight matrix for each position.
    #
    #       a) positional_hidden_layer_size == 0: No positional hidden layer;
    #          a separate logits layer per position.
    #
    #       b) positional_hidden_layer_size > 0: A single hidden layer per
    #          position, followed by a shared logits layer.
    #
    #   3) decode_strategy = "biased": The encoded sentence has a separate
    #      additive bias per position (equivalent to concatenating with
    #      position-specific embedding).
    #
    #       a) biased_hidden_layer_sizes == [0]: No hidden layers; a logits
    #          layer with position-specific biases.
    #
    #       b) biased_hidden_layer_sizes != [0]: A hidden layer with
    #          position-specific biases, followed by other hidden layers,
    #          followed by a shared logits layer.
    #
    vocab_size = self.config.vocab_size
    with tf.variable_scope("decoder"):

      if self.config.decode_strategy == "conditional":
        K = self.config.condition_length

        # Context vectors for both decode_pre and decode_post
        def _window_sum_3d(i, tensor):
          pre_sum = tf.reduce_sum(tensor[:, tf.maximum(i-K, 0):i, :], axis=1)
          post_sum = tf.reduce_sum(tensor[:, i+1:i+K+1, :], axis=1)
          if self.config.condition_uni_context:
            return pre_sum
          else:
            return pre_sum + post_sum

        def _window_sum_2d(i, tensor):
          pre_sum = tf.reduce_sum(tensor[:, tf.maximum(i-K, 0):i], axis=1)
          post_sum = tf.reduce_sum(tensor[:, i+1:i+K+1], axis=1)
          if self.config.condition_uni_context:
            return pre_sum
          else:
            return pre_sum + post_sum

        def _aggregate_context_vectors(decode_emb, decode_mask, offset):
          # Mask out padded embeddings to 0.
          masked_decode_emb = decode_emb * tf.to_float(
              tf.expand_dims(decode_mask, axis=2))

          # Aggregate over sliding window.
          context_vectors = tf.transpose(
              tf.map_fn(
                  lambda i: _window_sum_3d(i, masked_decode_emb),
                  tf.range(offset, offset + self.config.context_window),
                  dtype=tf.float32),
              (1, 0, 2))
          context_lengths = tf.transpose(
              tf.map_fn(
                  lambda i: _window_sum_2d(i, decode_mask),
                  tf.range(offset, offset + self.config.context_window),
                  dtype=tf.int32),
              (1, 0))
          context_mask = tf.where(
              tf.equal(context_lengths, 0),
              tf.zeros_like(context_lengths),
              tf.ones_like(context_lengths))
          context_lengths = tf.where(
              tf.equal(context_lengths, 0),
              tf.ones_like(context_lengths),
              context_lengths)

          context_vectors /= tf.to_float(
              tf.expand_dims(context_lengths, axis=2))

          return context_vectors, context_mask

        context_vectors_pre, context_pre_mask = _aggregate_context_vectors(
            self.decode_pre_emb,
            self.decode_pre_mask,
            K)
        self.context_vectors_pre = context_vectors_pre
        self.context_pre_mask = context_pre_mask
        context_vectors_post, context_post_mask = _aggregate_context_vectors(
            self.decode_post_emb,
            self.decode_post_mask,
            0)
        self.context_vectors_post = context_vectors_post
        self.context_post_mask = context_post_mask

        # Combine contexts and project to thought vector space
        contexts = tf.reshape(
            tf.concat([context_vectors_pre, context_vectors_post], axis=1),
            [-1, self.config.word_embedding_dim])
        contexts_mask = tf.reshape(
            tf.concat([context_pre_mask, context_post_mask], axis=1),
            [-1])
        self.contexts_concat = contexts
        self.contexts_concat_mask = contexts_mask
        if (self.config.logit_metric == "hyperbolic" and
            self.config.reparameterization == "independent"):
          num_units = self.config.encoder_dim + 1
        else:
          num_units = self.config.encoder_dim
        context_initializer = tf.random_uniform_initializer(
            minval=-1./self.config.encoder_dim,
            maxval=1./self.config.encoder_dim)
        contexts_projected = tf.contrib.layers.fully_connected(
            inputs=contexts,
            num_outputs=num_units,
            activation_fn=None,
            weights_initializer=context_initializer,
            scope="context_projection")
        self.contexts_projected = contexts_projected

        # Combine with thought_vectors and compute logits
        if self.config.logit_metric is None:
          thought_vectors = tf.reshape(
              tf.tile(
                  tf.expand_dims(self.thought_vectors, axis=1),
                  (1, 2 * self.config.context_window, 1)),
              [-1, self.config.encoder_dim])
          contexts_input = contexts_projected * tf.to_float(
              tf.expand_dims(contexts_mask, axis=1))
          fc = tf.contrib.layers.fully_connected(
              inputs=contexts_input + thought_vectors,
              num_outputs=vocab_size,
              activation_fn=None,
              scope="logits")
          self.local_context_vectors = contexts_projected

        else:
          # Denote B = batch_size, V = vocabulary_size
          if (self.config.logit_metric == "hyperbolic" and
              self.config.reparameterization == "independent"):
            num_units = self.config.encoder_dim + 1
          else:
            num_units = self.config.encoder_dim

          if self.config.logit_metric == "cosine":
            output_embeddings = tf.get_variable(
                "output_embeddings",
                shape=[vocab_size, num_units],
                initializer=self.uniform_initializer)
            # Similarities
            thought_vector_similarities = metrics.batched_cosine(
                self.thought_vectors,
                output_embeddings) # B x V
            context_vector_similarities = metrics.batched_cosine(
                contexts_projected,
                output_embeddings) # B x V
            self.output_embeddings = output_embeddings
            self.local_context_vectors = contexts_projected

          elif self.config.logit_metric == "hyperbolic":
            # Reparameterize so that norm is always less than 1
            if self.config.reparameterization == "tanh":
              output_embeddings = tf.get_variable(
                  "output_embeddings",
                  shape=[vocab_size, num_units],
                  initializer=self.uniform_initializer)
              scaling_factor = 1./tf.sqrt(float(self.config.encoder_dim))
              W_output = scaling_factor * tf.tanh(output_embeddings)
              contexts_rescaled = scaling_factor * tf.tanh(contexts_projected)

            elif self.config.reparameterization == "gaussian":
              output_embeddings = tf.get_variable(
                  "output_embeddings",
                  shape=[vocab_size, num_units],
                  initializer=self.uniform_initializer)
              output_temperature = 0.3 / (self.config.encoder_dim *
                                          (self.config.uniform_init_scale ** 2))
              W_output, W_scales = _gaussian_norm_scaling(output_embeddings,
                                                          output_temperature)
              context_temperature = (0.9 * self.config.condition_length /
                                     (self.config.uniform_init_scale ** 2))
              contexts_rescaled, context_scales = _gaussian_norm_scaling(
                  contexts_projected, context_temperature)

            elif self.config.reparameterization == "independent":
              output_embeddings = tf.get_variable(
                  "output_embeddings",
                  shape=[vocab_size, num_units],
                  initializer=self.uniform_initializer)
              W_output, W_scales = _independent_norm_scaling(
                  output_embeddings, tf.sigmoid,
                  self.config.independent_norm_shift)
              contexts_rescaled, context_scales = _independent_norm_scaling(
                  contexts_projected, tf.sigmoid,
                  self.config.independent_norm_shift)

            elif self.config.reparameterization == "projection":
              output_embeddings = tf.get_variable(
                  "output_embeddings",
                  shape=[vocab_size, num_units],
                  initializer=tf.random_uniform_initializer(
                      minval=-1./self.config.encoder_dim,
                      maxval=1./self.config.encoder_dim)
              )
              W_output = _project_to_unit_sphere(output_embeddings)
              W_scales = tf.norm(W_output, axis=1)
              contexts_rescaled = _project_to_unit_sphere(contexts_projected)
              context_scales = tf.norm(contexts_rescaled, axis=1)
            else:
              raise ValueError("Unrecognized reparameterization: %s" %
                               self.config.reparameterization)

            self.output_embeddings = W_output
            self.local_context_vectors = contexts_rescaled

            tf.summary.scalar("vector_norms/words/max", tf.reduce_max(
                tf.norm(W_output, axis=1)))
            tf.summary.scalar("vector_norms/context/max", tf.reduce_max(
                tf.norm(contexts_rescaled, axis=1)))
            tf.summary.scalar("vector_norms/thoughts/max", tf.reduce_max(
                tf.norm(self.thought_vectors, axis=1)))
            tf.summary.scalar("vector_norms/words/mean", tf.reduce_mean(
                tf.norm(W_output, axis=1)))
            tf.summary.scalar("vector_norms/context/mean", tf.reduce_mean(
                tf.norm(contexts_rescaled, axis=1)))
            tf.summary.scalar("vector_norms/thoughts/mean", tf.reduce_mean(
                tf.norm(self.thought_vectors, axis=1)))

            # Compute similarities
            thought_vector_similarities = - metrics.batched_hyperbolic(
                self.thought_vectors,
                W_output) # B x V
            context_vector_similarities = - metrics.batched_hyperbolic(
                contexts_rescaled,
                W_output) # 2BK x V

          context_vector_similarities = tf.reshape(
              tf.to_float(tf.expand_dims(contexts_mask, axis=1)) *
              context_vector_similarities,
              [-1, 2 * self.config.context_window, vocab_size]) # B x 2K x V


          # Scaling weights to control the trade-off between thought vector and
          # context vector similarities.
          lambda_1 = tf.get_variable(
              "thought_vector_weight",
              initializer=self.config.softmax_weights_initializer,
              dtype=tf.float32)
          lambda_2 = tf.get_variable(
              "context_vector_weight",
              initializer=self.config.softmax_weights_initializer,
              dtype=tf.float32)
          tf.summary.scalar("thought_vector_weight", lambda_1)
          tf.summary.scalar("context_vector_weight", lambda_2)
          self.thought_sims = thought_vector_similarities
          self.context_sims = context_vector_similarities

          # Compute logits
          if not self.config.debug_mode:
            logits = (
                lambda_1 * tf.expand_dims(thought_vector_similarities, axis=1) +
                lambda_2 * context_vector_similarities)
            # Bias
            biases = tf.get_variable(
                "biases",
                shape=[1, 1, vocab_size],
                initializer=tf.zeros_initializer()) # 1 x 1 x V
            logits = logits + biases
            # Broadcast
            fc = tf.reshape(logits, [-1, vocab_size]) # 2BK x V
          else:
            thought_vectors = tf.reshape(
                tf.tile(
                    tf.expand_dims(self.thought_vectors, axis=1),
                    (1, 2 * self.config.context_window, 1)),
                [-1, self.config.encoder_dim])
            contexts_input = contexts_rescaled * tf.to_float(
                tf.expand_dims(contexts_mask, axis=1))
            fc = tf.contrib.layers.fully_connected(
                inputs=lambda_1 * contexts_input + lambda_2 * thought_vectors,
                num_outputs=vocab_size,
                activation_fn=None,
                scope="logits")

        fc_is_logits = True

      elif self.config.decode_strategy == "positional":
        fc_is_logits = not self.config.positional_hidden_layer_size
        layer_size = self.config.positional_hidden_layer_size or vocab_size
        layer_scope = "logits" if fc_is_logits else "fully_connected"

        if fc_is_logits:
          # Loop over each softmax to avoid big parameter matrices
          fc_list = []
          for i in range(2 * self.config.context_window):
            fc_list.append(tf.contrib.layers.fully_connected(
                inputs=self.thought_vectors,
                num_outputs=layer_size,
                activation_fn=None,
                scope=layer_scope + "_%d" % i))
          fc = tf.concat(fc_list, 1)
        else:
          fc = tf.contrib.layers.fully_connected(
              inputs=self.thought_vectors,
              num_outputs=2 * self.config.context_window * layer_size,
              activation_fn=None,
              scope=layer_scope)
        fc = tf.reshape(fc, [-1, layer_size])
        if (not fc_is_logits) and self.config.positional_nonlinearity:
          fc = tf.nn.relu(fc)

      elif self.config.decode_strategy == "biased":
        layer_sizes = self.config.biased_hidden_layer_sizes
        fc_is_logits = not layer_sizes[0]
        first_layer_size = layer_sizes[0] or vocab_size
        first_layer_scope = "logits" if fc_is_logits else "fully_connected_1"

        embedding_projection = tf.contrib.layers.fully_connected(
            inputs=self.thought_vectors,
            num_outputs=first_layer_size,
            activation_fn=None,
            biases_initializer=None,  # Biases would be redundant.
            scope=first_layer_scope)

        # Matrix of position-specific word biases.
        with tf.variable_scope(first_layer_scope):
          biases = tf.get_variable(
              "biases",
              shape=[2 * self.config.context_window, first_layer_size],
              initializer=tf.zeros_initializer())

        # Broadcast-add biases to weights, and reshape to the desired
        # dimensions above.
        fc = tf.reshape(
            tf.expand_dims(embedding_projection, 1) + biases,
            [-1, first_layer_size])

        # Non-linearity
        fc = tf.nn.relu(fc)

        # Possibly more hidden layers.
        if layer_sizes:
          for i in range(1, len(layer_sizes)):
            fc = tf.contrib.layers.fully_connected(
                inputs=fc,
                num_outputs=layer_sizes[i],
                activation_fn=None,
                scope="fully_connected_%s" % (i + 1))
            fc = tf.nn.relu(fc)

      else:
        raise ValueError(
            "Invalid decode_strategy: %s" % self.config.decode_strategy)
      tf.summary.scalar("decoder_activations", tf.reduce_mean(fc))

      # logits has shape [batch_size * 2 * context_window, vocab_size]:
      # [
      #   [ batch index 0: decode_pre word 0  ]
      #   [ batch index 0: decode_pre word 1  ]
      #   ...
      #   [ batch index 0: decode_post word 0 ]
      #   [ batch index 0: decode_post word 1 ]
      #   ...
      #   [ batch index 1: decode_pre word 0  ]
      #   ...
      # ]
      if fc_is_logits:
        logits = fc
      else:
        logits = tf.contrib.layers.fully_connected(
            inputs=fc,
            num_outputs=vocab_size,
            activation_fn=None,
            scope="logits")
      self.logits = logits
      self.targets = targets
      self.target_mask = target_weights

    if (self.config.decode_strategy == "conditional" and
        self.config.logit_metric is None):
      with tf.variable_scope("decoder", reuse=True):
        self.output_embeddings = tf.transpose(
            tf.get_variable("logits/weights"), perm=(1, 0))

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)
    batch_loss = tf.reduce_sum(losses * target_weights)
    self.batch_loss = batch_loss
    tf.losses.add_loss(batch_loss)
    tf.summary.scalar("losses/batch_loss", batch_loss)

    # Correct predictions
    predictions = tf.argmax(logits, axis=1)
    correct = tf.to_float(tf.equal(predictions, targets))
    batch_acc = tf.reduce_sum(correct * target_weights) / tf.reduce_sum(
        target_weights)
    tf.summary.scalar("batch_accuracy", batch_acc)

    self.target_cross_entropy_losses.append(losses)
    self.target_cross_entropy_loss_weights.append(target_weights)
