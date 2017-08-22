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

"""Helper functions to convert Numpy arrays to TFRecords.

Use these functions if you have extracted floating point features for your
dataset in one numpy array X, and associated integer labels in another numpy
array Y. These are converted into TensorFlow Example protos (one for each row
in X) with the following fields:

  features: List of float32 / float64 features.
  label: Int64 label id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def _float_feature(values):
  """Helper for creating a Float Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(value):
  """Helper for creating an Int64 Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _create_serialized_example(features, label):
  """Helper for creating a serialized Example proto."""
  example = tf.train.Example(features=tf.train.Features(feature={
      "features": _float_feature(features),
      "label": _int64_feature(label),
  }))

  return example.SerializeToString()


def convert_to_tfrecord(features_array, labels_array, filename):
  """Convert to TFRecords and save in filename.

  Args:
    features_array: 2D Numpy array in float32 or float64 format. Each row
      contains features for one example.
    labels_array: 1D Numpy array of size features_array.shape[0]. Integer
      labels for the rows in features_array.
    filename: Location where to save resulting TFRecords.

  Raises:
    ValueError: If features_array is not float32 / float64, or if labels_array
      is not int32 / int64.
  """
  features_dtype = features_array.dtype
  labels_dtype = labels_array.dtype
  if features_dtype != np.float32 and features_dtype != np.float64:
    raise ValueError("Features must be float32 / float64. Found %s." %
                     features_dtype)
  if labels_dtype != np.int32 and labels_dtype != np.int64:
    raise ValueError("Labels must be int32 / int64. Found %s." % labels_dtype)

  assert features_array.shape[0] == labels_array.shape[0], (
      "Features shape != Labels shape. %d != %d" % (features_array.shape[0],
                                                    labels_array.shape[0]))

  with tf.python_io.TFRecordWriter(filename) as writer:
    for i in range(features_array.shape[0]):
      serialized = _create_serialized_example(features_array[i, :],
                                              labels_array[i])
      writer.write(serialized)
