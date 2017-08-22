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

"""Model and training configurations for skip-thoughts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def base_model_config():
  """Returns the default configuration for a skip-thoughts model, as a dict."""
  return {
      # TFRecord file pattern containing Example protos.
      "input_file_pattern": "",

      # Number of examples to keep in the input queue.
      "input_queue_capacity": 5 * 640000,  # 5 shards of the BookCorpus.

      # Number of threads for prefetching TFRecord values.
      "num_input_reader_threads": 1,

      # Number of threads for deserializing Example protos and filling the batch
      # queue.
      "num_batching_threads": 1,

      # Whether to shuffle the input data.
      "shuffle_input_data": True,

      # Scale of the random uniform initializer.
      "uniform_init_scale": 0.1,

      # Number of unique words in the vocab.
      "vocab_size": 20000,

      # Batch size (training and evaluation only).
      "batch_size": 128,

      # Word embedding dimension.
      "word_embedding_dim": 620,

      # Whether to use a bidirectional or unidirectional encoder RNN.
      "bidirectional_encoder": False,

      # Number of output dimensions of the sentence encoder.
      "encoder_dim": 2400,

      # Operation for combining the output GRU states from encoder
      "pooling_operation": "last",

      # Number of words on either side of the sentence to use as the context.
      "context_window": 3,

      # Run in debug mode.
      "debug_mode": False,

      # "conditional": the local context around the target word is used during
      # prediction.
      # "positional": the encoded sentence is multiplied by a separate weight
      # matrix for each position.
      # "biased": the encoded sentence has a separate additive bias per position
      # (equivalent to concatenating with position-specific embedding).
      "decode_strategy": "biased",

      # If decode_strategy == "conditional":
      # Number of local context tokens to condition the prediction on.
      "condition_length": 1,

      # If decode_strategy == "conditional":
      # Whether to use only context before, or both context before and after the
      # token to be predicted.
      "condition_uni_context": True,

      # If decode_strategy == "conditional":
      # Metric to use for computing logits.
      "logit_metric": None,

      # If decode_strategy == "conditional" and logit_metric is not None:
      # Initial value of weight multipliers for softmax.
      "softmax_weights_initializer": 10.,

      # If decode_strategy == "conditional" and logit_metric == "hyperbolic":
      # Reparameterization method and temperature of the gaussian, scaling
      # factor for independent norms.
      "reparameterization": "gaussian",
      "gaussian_temperature": 1.,
      "independent_norm_shift": -2.,

      # If decode_strategy == "positional":
      # Size of the separate positional hidden layers before a shared softmax
      # decoder. The default value of 0 denotes no hidden layer and positional
      # softmax decoders.
      "positional_hidden_layer_size": 0,

      # If decode_strategy == "positional" and positional_hidden_layer_size>0:
      # Whether to use nonlinearity after the hidden positional layers.
      "positional_nonlinearity": False,

      # If decode_strategy == "biased":
      # List of fully connected hidden layer sizes between the sentence encoder
      # and the softmax decoder. The default value of [0] denotes no hidden
      # layers (tf.HParams does not allow empty hyperparameter lists).
      "biased_hidden_layer_sizes": [0],
  }


def base_training_config():
  """Returns the default configuration for training, as a dict."""
  return {
      # Name of the gradient optimizer. See ops/training.py.
      "optimizer": "adam",

      # Optimizer-specific parameters.
      "momentum": 0.9,  # For momentum optimizer.
      "adam_beta1": 0.9,  # For adam optimizer.
      "adam_beta2": 0.999,  # For adam optimizer.
      "adam_epsilon": 1e-08,  # For adam optimizer.

      # Initial learning rate.
      "learning_rate": 0.0008,

      # If > 0, the learning rate decay factor.
      "learning_rate_decay_factor": 0.5,

      # The number of steps before the learning rate decays by
      # learning_rate_decay_factor.
      "learning_rate_decay_steps": 400000,

      # If True, decay the learning rate at discrete intervals.
      "learning_rate_decay_staircase": False,

      # The minimum value to decay the learning rate to.
      "learning_rate_decay_floor": 0,

      # If > 0, the number of training steps.
      "number_of_steps": 0,

      # If > 0, clip gradients to this value.
      "clip_gradient_norm": 5.0,

      # How often (in seconds) to save model checkpoints.
      "save_model_secs": 60 * 10,

      # How often (in hours) checkpoints should be kept.
      "keep_checkpoint_every_n_hours": 2,

      # How often (in seconds) to save model summaries.
      "save_summaries_secs": 60 * 10,

      # How many model checkpoints to keep.
      "max_checkpoints_to_keep": 5,

      # Startup delay between worker replicas and chief. Only applies for async
      # multi-worker training.
      "startup_delay_steps": 100,
  }


def _override(config, overrides):
  """Overrides base configuration parameters.

  Args:
    config: Configuration dict.
    overrides: Dict of override parameter names to values.

  Raises:
    KeyError: If an unrecognized parameter is passed.
  """
  for key, value in overrides.iteritems():
    if key not in config:
      raise KeyError("Unrecognized parameter: %s" % key)
    config[key] = value


def model_config(**overrides):
  """Creates a model configuration object.

  Args:
    **overrides: Key-value pairs where the key is the parameter name and the
      value is the value for the parameter.

  Returns:
    A tf.contrib.training.HParams object.
  """
  config = base_model_config()
  _override(config, overrides)
  return tf.contrib.training.HParams(**config)


def training_config(**overrides):
  """Creates a training configuration object.

  Args:
    **overrides: Key-value pairs where the key is the parameter name and the
      value is the value for the parameter.

  Returns:
    A tf.contrib.training.HParams object.
  """
  config = base_training_config()
  _override(config, overrides)
  return tf.contrib.training.HParams(**config)
