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

"""Build word_context_model and inspect intermediate tensors for sample inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import numpy as np
import tensorflow as tf

from skip_thoughts_dist.ops import training
from word_context import configuration
from word_context import word_context_model


def main(argv):
  # Set up the model config.
  overrides = {
    "vocab_size": 5,
    "batch_size": 2,
    "input_file_pattern": "/path/to/tfrecord/files",
    "word_embedding_dim": 2,
    "encoder_dim": 2,
    "decode_strategy": "conditional",
    "logit_metric": "hyperbolic",
    "reparameterization": "independent",
    "softmax_weights_initializer": 10.0,
    #"independent_norm_scaling": -2,
  }
  model_config = configuration.model_config(**overrides)
  tf.logging.info("model_config: %s",
                  json.dumps(model_config.values(), indent=2))
  # Set up the training config
  overrides = {
    "optimizer": "adam",
    "learning_rate": 0.0005,
  }
  training_config = configuration.training_config(**overrides)
  tf.logging.info("training_config: %s",
                  json.dumps(training_config.values(), indent=2))

  tf.logging.info("Building training graph.")
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    tf.logging.info("Building model.")
    model = word_context_model.WordContextModel(
        model_config, mode="train")
    model.build()

    # Set up the learning rate and optimizer.
    learning_rate = training.create_learning_rate(training_config,
                                                  model.global_step)
    optimizer = training.create_optimizer(training_config, learning_rate)

    train_tensor = training.create_train_op(training_config, optimizer, model)


  feed_dict = {
      model.encode_ids: np.random.randint(0, model_config.vocab_size,
                                          size=(2, 3)),
      model.encode_mask: np.asarray([[1, 1, 0], [1, 0, 0]], dtype="int32"),
      model.decode_pre_ids: np.random.randint(0, model_config.vocab_size,
                                          size=(2, 4)),
      model.decode_pre_mask: np.asarray([[0, 1, 1, 0], [0, 1, 0, 0]], dtype="int32"),
      model.decode_post_ids: np.random.randint(0, model_config.vocab_size,
                                          size=(2, 4)),
      model.decode_post_mask: np.asarray([[1, 1, 0, 0], [1, 0, 0, 0]], dtype="int32"),
  }


  fetch_tensors = [
      model.encode_ids,
      model.decode_pre_ids,
      model.decode_post_ids,
      model.encode_emb,
      model.decode_pre_emb,
      model.decode_post_emb,
      model.thought_vectors,
      model.encoder_outputs,
      model.context_vectors_pre,
      model.context_pre_mask,
      model.context_vectors_post,
      model.context_post_mask,
      model.contexts_concat,
      model.contexts_projected,
      model.contexts_concat_mask,
      model.local_context_vectors,
      model.output_embeddings,
      model.W_output,
      model.contexts_rescaled,
      model.context_scales,
      model.thoughts_rescaled,
      model.thought_sims,
      model.context_sims,
      model.logits,
      model.batch_loss,
      train_tensor,
  ]

  with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1):
      for var in tf.trainable_variables():
        var_eval = var.eval()
        print(var.name)
        print(var_eval)

      outputs = sess.run(fetch_tensors, feed_dict=feed_dict)
      for ii, item in enumerate(fetch_tensors):
        if isinstance(item, str):
          print(item)
        else:
          print(item.name)
        print(outputs[ii].shape)
        print(outputs[ii])
        print()


if __name__ == '__main__':
  tf.app.run()
