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

"""Class for encoding text using a trained WordContextModel.

This class inherits from SkipThoughtsEncoder class.

Example usage:
  g = tf.Graph()
  with g.as_default():
    encoder = WordContextEncoder(embeddings)
    restore_fn = encoder.build_graph_from_config(model_config, checkpoint_path)

  with tf.Session(graph=g) as sess:
    restore_fn(sess)
    thought_vectors = encoder.encode(sess, data)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from skip_thoughts import skip_thoughts_encoder
from word_context import word_context_model


class WordContextEncoder(skip_thoughts_encoder.SkipThoughtsEncoder):
  """Word-context sentence encoder."""

  def __init__(self, embeddings):
    """Initializes the encoder.

    Args:
      embeddings: Dictionary of word to embedding vector (1D numpy array).
    """
    super(WordContextEncoder, self).__init__(embeddings)

  def build_graph_from_config(self, model_config, checkpoint_path):
    """Builds the inference graph from a configuration object.

    Args:
      model_config: Object containing configuration for building the model.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    tf.logging.info("Building model.")
    model = word_context_model.WordContextModel(model_config, mode="encode")
    model.build()
    saver = tf.train.Saver()

    return self._create_restore_fn(checkpoint_path, saver)
