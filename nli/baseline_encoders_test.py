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

"""Tests for nli.baseline_encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from nli import baseline_encoders


class BaselineEncodersTest(unittest.TestCase):

  def testRandomEncoder(self):
    data = ["Hello World!",
            "Is there a brain in google or a google in the brain?"]
    dim = 20
    encoder = baseline_encoders.RandomEncoder(dim)
    vectors = encoder.encode(data)
    self.assertEqual((dim,), vectors[0].shape)
    self.assertEqual(len(vectors), 2)

if __name__ == "__main__":
  unittest.main()
