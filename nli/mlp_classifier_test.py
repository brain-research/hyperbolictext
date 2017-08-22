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

"""Tests for nli.mlp_classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np

from nli import mlp_classifier
from nli import tfrecord_creator


class MlpClassifierTest(unittest.TestCase):

  def testLinearClasses(self):
    test_subdirectory = os.path.join(os.path.dirname(__file__),
                                     'mlp_test_linear')
    os.makedirs(test_subdirectory)

    # Generate linearly separable data for two classes and divide into train
    # and dev sets
    features_1 = np.random.uniform(low=1., high=2., size=(100, 10))
    features_2 = np.random.uniform(low=-2., high=-1., size=(100, 10))
    train_features = np.vstack([features_1[:90, :], features_2[:90, :]])
    train_labels = np.asarray([0]*90 + [1]*90)
    dev_features = np.vstack([features_1[90:, :], features_2[90:, :]])
    dev_labels = np.asarray([0]*10 + [1]*10)

    train_filename = os.path.join(test_subdirectory, 'train_data.tfrecord')
    tfrecord_creator.convert_to_tfrecord(train_features, train_labels,
                                         train_filename)
    dev_filename = os.path.join(test_subdirectory, 'dev_data.tfrecord')
    tfrecord_creator.convert_to_tfrecord(dev_features, dev_labels, dev_filename)

    mlp = mlp_classifier.MLPClassifier(10, 2, test_subdirectory, depth=1,
                                       hidden_size=10)
    num_updates = mlp.fit(train_filename, dev_filename, logging_frequency=1)
    self.assertGreater(num_updates, 0)
    loss, accuracy, outputs = mlp.predict(dev_filename)

    self.assertLess(loss, 1e8)
    self.assertGreaterEqual(accuracy, 0.)
    self.assertLessEqual(accuracy, 1.)
    self.assertEqual(outputs.shape[0], 20)
    self.assertEqual(outputs.shape[1], 2)
    self.assertTrue((outputs <= 1.).all())
    self.assertTrue((outputs >= 0.).all())

if __name__ == '__main__':
  unittest.main()
