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

"""Tests for word_context.metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from word_context import metrics


class MetricsTest(tf.test.TestCase):

  def testCheckHyperbolic(self):
    tensor_1 = tf.constant([[.1, .2],
                            [0., 0.],
                            [1., 0.]],
                           dtype=tf.float32)
    tensor_2 = tf.constant([[0., 1.],
                            [.1, .2]],
                           dtype=tf.float32)

    with self.test_session() as sess:
      distances = sess.run(
          metrics.batched_hyperbolic(tensor_1, tensor_2))

    self.assertFalse(np.isnan(distances).any())
    self.assertFalse(np.isinf(distances).any())
    self.assertAlmostEqual(distances[1,1], 0.454899, places=6)
    self.assertAlmostEqual(distances[0,1], 0.0, places=6)

  def testCheckCosine(self):
    tensor_1 = tf.constant([[10., .2],
                            [0., 0.],
                            [1., 0.]],
                           dtype=tf.float32)
    tensor_2 = tf.constant([[0., 0.],
                            [.1, .2]],
                           dtype=tf.float32)

    with self.test_session() as sess:
      distances = sess.run(
          metrics.batched_cosine(tensor_1, tensor_2))

    self.assertFalse(np.isnan(distances).any())
    self.assertFalse(np.isinf(distances).any())
    self.assertAlmostEqual(distances[0,1], 0.465009, places=6)
    self.assertAlmostEqual(distances[0,0], 0.0, places=6)
    self.assertAlmostEqual(distances[1,0], 0.0, places=6)


if __name__ == '__main__':
  tf.test.main()
