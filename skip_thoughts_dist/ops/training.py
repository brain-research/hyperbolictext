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

"""Ops for training a skip-thought vectors model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def create_learning_rate(config, global_step):
  """Sets up the learning rate with optional exponential decay.

  Args:
    config: Object containing learning rate configuration parameters.
    global_step: Tensor; the global step.

  Returns:
    learning_rate: Tensor; the learning rate with exponential decay.
  """
  if config.learning_rate_decay_factor > 0:
    learning_rate = tf.train.exponential_decay(
        learning_rate=float(config.learning_rate),
        global_step=global_step,
        decay_steps=config.learning_rate_decay_steps,
        decay_rate=config.learning_rate_decay_factor,
        staircase=config.learning_rate_decay_staircase)

    if config.learning_rate_decay_floor > 0:
      learning_rate = tf.maximum(learning_rate,
                                 config.learning_rate_decay_floor)
  else:
    learning_rate = tf.constant(config.learning_rate)
  tf.summary.scalar("learning_rate", learning_rate)
  return learning_rate


def create_optimizer(config, learning_rate):
  """Creates an Optimizer for training.

  Args:
    config: Object containing optimizer configuration parameters.
    learning_rate: Tensor; the learning rate.

  Returns:
    optimizer: An Optimizer for training.

  Raises:
    ValueError: If config.optimizer is unrecognized.
  """
  if config.optimizer == "momentum":
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=config.momentum)
  elif config.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif config.optimizer == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif config.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon)
  elif config.optimizer == "rmsprop":
    optimizer = tf.RMSPropOptimizer(learning_rate)
  else:
    raise ValueError("Unknown optimizer: %s" % config.optimizer)

  return optimizer


def create_train_op(config, optimizer, model):
  """Creates a Tensor to train the model.

  Args:
    config: Object containing training configuration parameters.
    optimizer: Instance of tf.train.Optimizer.
    model: Model to train.

  Returns:
    A Tensor to run a single step of training. The value of the Tensor is the
    total loss on the training batch.
  """

  def transform_grads_fn(grads):
    # Clip gradients.
    if config.clip_gradient_norm > 0:
      with tf.name_scope("clip_grads"):
        grads = tf.contrib.training.clip_gradient_norms(
            grads, config.clip_gradient_norm)
    return grads

  return tf.contrib.training.create_train_op(
      total_loss=model.total_loss,
      optimizer=optimizer,
      global_step=model.global_step,
      transform_grads_fn=transform_grads_fn)
