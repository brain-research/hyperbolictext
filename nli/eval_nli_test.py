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

"""Tests for nli.eval_nli."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

from nli import eval_nli
from nli.baseline_encoders import RandomEncoder

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__),
                                 'testdata', 'sample')


class EvalMultinliTest(unittest.TestCase):

  def testRandomAccuracyLogistic(self):
    dim = 20
    encoder = RandomEncoder(dim)
    results = eval_nli.evaluate(encoder, TESTDATA_FILENAME, None)
    # Std err of random predictions for small n is too high to say anything
    # about the accuracy. Just check if these lie between 0 and 1 for now.
    for k, v in results.iteritems():
      if k == 'best_hyperparameters':
        self.assertIn(v, results)
      else:
        dev_result = v['dev']
        test_result = v['test']
        self.assertIn('overall', dev_result)
        self.assertGreaterEqual(dev_result['overall'], 0.)
        self.assertLessEqual(dev_result['overall'], 1.)
        self.assertIn('overall', test_result)
        self.assertGreaterEqual(test_result['overall'], 0.)
        self.assertLessEqual(test_result['overall'], 1.)

  def testRandomAccuracyMLP(self):
    dim = 20
    encoder = RandomEncoder(dim)
    test_subdirectory = os.path.join(os.path.dirname(__file__),
                                     'mlp_test_random')
    os.makedirs(test_subdirectory)
    results = eval_nli.evaluate(
        encoder, TESTDATA_FILENAME, test_subdirectory, method='mlp')
    # Std err of random predictions for small n is too high to say anything
    # about the accuracy. Just check if these lie between 0 and 1 for now.
    for k, v in results.iteritems():
      if k == 'best_hyperparameters':
        self.assertIn(v, results)
      else:
        dev_result = v['dev']
        test_result = v['test']
        self.assertIn('overall', dev_result)
        self.assertGreaterEqual(dev_result['overall'], 0.)
        self.assertLessEqual(dev_result['overall'], 1.)
        self.assertIn('overall', test_result)
        self.assertGreaterEqual(test_result['overall'], 0.)
        self.assertLessEqual(test_result['overall'], 1.)

if __name__ == '__main__':
  unittest.main()
