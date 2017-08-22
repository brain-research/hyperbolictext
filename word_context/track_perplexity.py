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

"""Track training progress via per-word perplexity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf

from skip_thoughts_dist.ops import evaluation
from word_context import configuration
from word_context import word_context_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("master", "", "Name of the TensorFlow master.")

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", None,
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", None, "Directory to write event logs to.")

tf.flags.DEFINE_integer("eval_interval_secs", 600,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 50000,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_string("model_config_overrides", "",
                       "JSON string containing configuration overrides.")

tf.flags.DEFINE_integer("min_global_step", 100,
                        "Minimum global step to compute perplexity.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  # Set up the model config.
  model_config = configuration.model_config(
      input_file_pattern=FLAGS.input_file_pattern,
      input_queue_capacity=FLAGS.num_eval_examples,
      shuffle_input_data=False)
  if FLAGS.model_config_overrides:
    model_config.parse_json(FLAGS.model_config_overrides)
  config_json = json.dumps(model_config.values(), indent=2)
  tf.logging.info("model_config: %s", config_json)

  with tf.Graph().as_default():
    # Build the model for evaluation.
    model = word_context_model.WordContextModel(
        model_config, mode="train")
    model.build()

    evaluation.evaluate_repeatedly(
        model=model,
        checkpoint_dir=FLAGS.checkpoint_dir,
        eval_dir=FLAGS.eval_dir,
        num_eval_examples=FLAGS.num_eval_examples,
        min_global_step_for_perplexity=FLAGS.min_global_step,
        master=FLAGS.master,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == "__main__":
  tf.app.run()
