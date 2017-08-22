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

"""Helper functions for different distance metrics.

TODO(bdhingra): DO NOT SUBMIT without a detailed description of hyperbolic.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

EPSILON = 1e-15
INFINITY = np.inf


def _squared_norm(x):
  """Computes squared L2 norm.

  Args:
    x: Tensor of arbitrary shape.

  Returns:
    norms: Squared L2 norm of x computed along last axis.
  """
  return tf.reduce_sum(tf.square(x), axis=-1)


def _arcosh(z):
  """Compute inverse hyperbolic cosine.

  Args:
    z: Tensor of arbitrary shape.

  Returns:
    Elementwise arcosh(z).
  """
  z_clip = tf.clip_by_value(z, 1., INFINITY)
  return tf.log(z_clip + tf.sqrt(tf.square(z_clip) - 1. + EPSILON))


def _hyperbolic_distance(norm_x, norm_y, norm_xy):
  """Compute hyperbolic distance between x and y.

  Args:
    norm_x: Squared norm of x.
    norm_y: Squared norm of y.
    norm_xy: Squared norm of x - y.

  Returns:
    Elementwise hyperbolic distance between x and y.
  """
  return _arcosh(1. + 2. * norm_xy /
                 tf.clip_by_value((1. - norm_x) * (1. - norm_y), EPSILON, 1.))


def batched_hyperbolic(A, B):
  """Compute pair-wise hyperbolic distance.

  Args:
    A: 2D Tensor.
    B: 2D Tensor such that B.shape[1] == A.shape[1].

  Returns:
    distances: Tensor with shape [A.shape[0], B.shape[0]], where
      distances[i,j] = hyperbolic_distance(A[i,:], B[j,:]).
  """
  # Denote A.shape[0] as M, B.shape[0] as N
  squared_norm_A = tf.tile(
      tf.expand_dims(_squared_norm(A), axis=1),
      [1, tf.shape(B)[0]]) # M x N
  squared_norm_B = tf.tile(
      tf.expand_dims(_squared_norm(B), axis=0),
      [tf.shape(A)[0], 1]) # M x N
  inner_product = tf.matmul(A, tf.transpose(B, (1,0))) # M x N

  squared_norm_AB = squared_norm_A + squared_norm_B - 2 * inner_product

  distances = _hyperbolic_distance(squared_norm_A,
                                   squared_norm_B,
                                   squared_norm_AB)

  return distances


def batched_cosine(A, B):
  """Compute pair-wise cosine similarity.

  Args:
    A: 2D Tensor.
    B: 2D Tensor such that B.shape[1] == A.shape[1].

  Returns:
    similarities: Tensor with shape [A.shape[0], B.shape[0]], where
      distances[i,j] = cosine_distance(A[i,:], B[j,:]).
  """
  # Denote A.shape[0] as M, B.shape[0] as N
  normalized_A = tf.nn.l2_normalize(A, dim=1) # M x d
  normalized_B = tf.nn.l2_normalize(B, dim=1) # N x d

  similarities = tf.matmul(normalized_A, tf.transpose(normalized_B, (1, 0)))

  return similarities
