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

"""Tests for nli.tfrecord_creator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np
import tensorflow as tf

from nli import tfrecord_creator


class TfrecordCreatorTest(unittest.TestCase):

  def testSaveAndLoad(self):
    test_filename = os.path.join(os.path.dirname(__file__),
                                 'dataset.tfrecords')
    features = np.random.normal(size=(3, 3))
    labels = np.random.randint(2, size=3)

    # Save
    tfrecord_creator.convert_to_tfrecord(features, labels, test_filename)

    # Load
    features_ = np.zeros((3, 3))
    labels_ = np.zeros((3,), dtype='int64')

    record_iterator = tf.python_io.tf_record_iterator(path=test_filename)
    for i, string_record in enumerate(record_iterator):
      example = tf.train.Example()
      example.ParseFromString(string_record)

      features_[i, :] = example.features.feature['features'].float_list.value
      labels_[i] = example.features.feature['label'].int64_list.value[0]

    self.assertTrue(np.isclose(features, features_).all())
    self.assertTrue((labels_ == labels).all())


if __name__ == '__main__':
  unittest.main()
