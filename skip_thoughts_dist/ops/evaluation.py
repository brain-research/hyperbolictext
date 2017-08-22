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

"""Ops for evaluating the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class _ComputeLengthNormsHook(tf.train.SessionRunHook):
  """Hook to compute average norm for different input lengths."""

  def __init__(self,
               thought_vector_tensor,
               mask_tensor,
               log_dir=None,
               summary_writer=None,
               min_global_step=None):
    """Initializes the ComputeLengthNormsHook monitor.

    Args:
      thought_vector_tensor: A Tensor of shape B x D, where B is batch size,
        and D is embedding dimension; output of the encoder GRU or LSTM at each
        time-step. Thought vectors for the batch.
      mask_tensor: A Tensor of shape B x N; masks specifying padding of the
        sequences (from which lengths can be computed).
      log_dir: Directory to save summary events to. Used only when
        summary_writer is not provided.
      summary_writer: A tf.summary.FileWriter to write summary events with.
      min_global_step: If not None, the minimum global step at which to compute
        perplexity. This is used to prevent computing perplexity at the start
        of training, when perplexity may be very large because it's exponential
        with respect to loss.

    Raises:
      ValueError: If neither log_dir nor summary_writer is provided.
    """
    self._thought_vector_tensor = thought_vector_tensor
    self._mask_tensor = mask_tensor
    self._log_dir = log_dir
    self._summary_writer = summary_writer
    if self._log_dir is None and self._summary_writer is None:
      raise ValueError("One of log_dir or summary_writer should be provided.")

    self._min_global_step = min_global_step
    self._global_step = tf.train.get_or_create_global_step()

  def begin(self):
    # Indicates whether global_step >= self._min_global_step.
    self._should_run = True

    # Accumulators over evaluation batches.
    self._length_sum_norms = {}
    self._length_counts = {}

    # Initialize the FileWriter.
    if self._summary_writer is None and self._log_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._log_dir)

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    global_step = tf.train.global_step(session, self._global_step)
    if self._min_global_step and global_step < self._min_global_step:
      tf.logging.info("Skipping perplexity evaluation: global step = %d < %d",
                      global_step, self._min_global_step)
      self._should_run = False

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._should_run:
      return tf.train.SessionRunArgs(
          [self._thought_vector_tensor, self._mask_tensor])

  def after_run(self, run_context, run_values):
    if self._should_run:
      thought_vectors, masks = run_values.results
      lengths = masks.sum(axis=1)

      # Compute norms
      thought_vector_norms = np.linalg.norm(thought_vectors, axis=1)

      # Bin by length
      for i in range(lengths.shape[0]):
        length = int(lengths[i])
        if length not in self._length_sum_norms:
          self._length_sum_norms[length] = 0.0
          self._length_counts[length] = 0
        self._length_sum_norms[length] += thought_vector_norms[i]
        self._length_counts[length] += 1

  def end(self, session):
    if self._should_run:
      for length in [1, 5, 10, 15, 20]:
        if length in self._length_sum_norms:
          average_norm = (self._length_sum_norms[length] /
                          self._length_counts[length])
          tf.logging.info("Length %d Average Norm = %.4f", length, average_norm)

          # Log to the SummaryWriter.
          if self._summary_writer:
            summary = tf.Summary()
            value = summary.value.add()
            value.simple_value = average_norm
            value.tag = "length_norms/average_norm_%d" % length
            global_step = tf.train.global_step(session, self._global_step)
            self._summary_writer.add_summary(summary, global_step)
            self._summary_writer.flush()


class _ComputePerplexityHook(tf.train.SessionRunHook):
  """Hook to compute per-word perplexity during evaluation."""

  def __init__(self,
               losses_tensor,
               weights_tensor,
               log_dir=None,
               summary_writer=None,
               min_global_step=None):
    """Initializes the ComputePerplexityHook monitor.

    Args:
      losses_tensor: A Tensor of any shape; the target cross entropy losses for
        the current batch.
      weights_tensor: A Tensor of weights corresponding to losses.
      log_dir: Directory to save summary events to. Used only when
        summary_writer is not provided.
      summary_writer: A tf.summary.FileWriter to write summary events with.
      min_global_step: If not None, the minimum global step at which to compute
        perplexity. This is used to prevent computing perplexity at the start
        of training, when perplexity may be very large because it's exponential
        with respect to loss.

    Raises:
      ValueError: If neither log_dir nor summary_writer is provided.
    """
    self._losses_tensor = losses_tensor
    self._weights_tensor = weights_tensor
    self._log_dir = log_dir
    self._summary_writer = summary_writer
    if self._log_dir is None and self._summary_writer is None:
      raise ValueError("One of log_dir or summary_writer should be provided.")

    self._min_global_step = min_global_step
    self._global_step = tf.train.get_or_create_global_step()

  def begin(self):
    # Indicates whether global_step >= self._min_global_step.
    self._should_run = True

    # Accumulators over evaluation batches.
    self._sum_losses = 0.0
    self._sum_weights = 0.0
    self._sum_correct = 0.0

    # Initialize the FileWriter.
    if self._summary_writer is None and self._log_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._log_dir)

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    global_step = tf.train.global_step(session, self._global_step)
    if self._min_global_step and global_step < self._min_global_step:
      tf.logging.info("Skipping perplexity evaluation: global step = %d < %d",
                      global_step, self._min_global_step)
      self._should_run = False

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._should_run:
      return tf.train.SessionRunArgs(
          [self._losses_tensor, self._weights_tensor])

  def after_run(self, run_context, run_values):
    if self._should_run:
      losses, weights = run_values.results
      self._sum_losses += np.sum(losses * weights)
      self._sum_weights += np.sum(weights)

  def end(self, session):
    if self._should_run and self._sum_weights > 0:
      perplexity = float(np.exp(self._sum_losses / self._sum_weights))
      tf.logging.info("Perplexity = %.4f", perplexity)

      # Log perplexity, accuracy, total loss to the SummaryWriter.
      if self._summary_writer:
        summary = tf.Summary()
        value = summary.value.add()
        value.simple_value = perplexity
        value.tag = "perplexity"
        value = summary.value.add()
        value.simple_value = self._sum_losses
        value.tag = "total_loss"
        global_step = tf.train.global_step(session, self._global_step)
        self._summary_writer.add_summary(summary, global_step)
        self._summary_writer.flush()


def evaluate_repeatedly(model,
                        checkpoint_dir,
                        eval_dir,
                        num_eval_examples,
                        min_global_step_for_perplexity=None,
                        master="",
                        eval_interval_secs=600):
  """Repeatedly searches for a checkpoint in checkpoint_dir and evaluates it.

  Args:
    model: A built instance of SkipThoughtsModel.
    checkpoint_dir: Directory containing model checkpoints.
    eval_dir: Directory to save summary events to.
    num_eval_examples: Number of examples for evaluation.
    min_global_step_for_perplexity: If not None, the minimum global step at
      which to compute perplexity. This is used to prevent computing perplexity
      at the start of training, when perplexity may be very large because it's
      exponential with respect to loss.
    master: Name of the TensorFlow master.
    eval_interval_secs: Interval (in seconds) between evaluation runs.
  """
  # Number of batches to evaluate.
  num_eval_batches = int(np.ceil(num_eval_examples / model.config.batch_size))

  losses_tensor = tf.concat(model.target_cross_entropy_losses, 0)
  weights_tensor = tf.concat(model.target_cross_entropy_loss_weights, 0)

  thought_vector_tensor = model.thought_vectors
  mask_tensor = model.encode_mask

  hooks = [
      # Run num_eval_batches iterations.
      tf.contrib.training.StopAfterNEvalsHook(num_eval_batches),
      # Save a summary at the end.
      tf.contrib.training.SummaryAtEndHook(log_dir=eval_dir),
      # Compute per-word perplexity over the evaluation set.
      _ComputePerplexityHook(
          losses_tensor,
          weights_tensor,
          log_dir=eval_dir,
          min_global_step=min_global_step_for_perplexity),
      # Compute average norms for different sequence lengths.
      _ComputeLengthNormsHook(
          thought_vector_tensor,
          mask_tensor,
          log_dir=eval_dir,
          min_global_step=min_global_step_for_perplexity),
  ]

  tf.contrib.training.evaluate_repeatedly(
      checkpoint_dir=checkpoint_dir,
      master=master,
      eval_ops=[losses_tensor, weights_tensor,
                mask_tensor, thought_vector_tensor],
      eval_interval_secs=eval_interval_secs,
      hooks=hooks)
