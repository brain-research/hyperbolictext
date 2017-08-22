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

"""Classes implementing baseline encoders for evaluation on MultiNLI.

This file holds classes implementing baselines such as Random, Average
Word Embeddings for encoding sentences to vector representations. These
are meant mainly as a test for the eval_multinli.py module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class RandomEncoder(object):
  """Returns a random vector for each input sentence."""

  def __init__(self, dim, mean=0., stddev=1.):
    """Create a random encoder.

    Args:
      dim: Dimensionality of encoded vectors.
      mean: Mean of the generating normal distribution along each dimension.
      stddev: Standard deviation of the generating normal distribution along
        each dimension.
    """
    self.dim = dim
    self.normal_mean = mean
    self.normal_std = stddev

  def encode(self, text_data):
    """Encodes the input sentences to random vectors.

    Args:
      text_data: list of strings

    Returns:
      vectors: A list of numpy arrays corresponding to the
        encodings of sentences in 'data'.
    """
    return [np.random.normal(loc=self.normal_mean,
                             scale=self.normal_std,
                             size=(self.dim,))
            for _ in range(len(text_data))]
