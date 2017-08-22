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

"""Train the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os.path

import tensorflow as tf

from skip_thoughts import skip_thoughts_model
from skip_thoughts_dist import configuration
from skip_thoughts_dist.ops import training

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("master", "", "Name of the TensorFlow master.")
tf.flags.DEFINE_integer("task", 0, "Task id of this worker replica.")
tf.flags.DEFINE_integer("ps_tasks", 0, "Number of parameter servers.")

tf.flags.DEFINE_boolean("sync_replicas", False,
                        "Whether to sync gradient updates between workers.")
tf.flags.DEFINE_integer("replicas_to_aggregate", 1,
                        "Number of gradients to collect before updating model "
                        "parameters.")
tf.flags.DEFINE_integer("total_num_replicas", 1,
                        "Total number of worker replicas.")

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of input SSTable files.")
tf.flags.DEFINE_string("train_dir", None,
                       "Directory for saving and loading checkpoints.")

tf.flags.DEFINE_string("model_config_overrides", "",
                       "JSON string containing configuration overrides.")
tf.flags.DEFINE_string("training_config_overrides", "",
                       "JSON string containing configuration overrides.")


def _log_config(config, name):
  """Logs a tf.HParams object.

  If this worker is the chief (i.e. FLAGS.task == 0), the config is additionally
  written as a JSON-serialized file.

  Args:
    config: tf.HParams object.
    name: Config name (e.g. "model_config").
  """
  config_json = json.dumps(config.values(), indent=2)
  tf.logging.info("%s: %s", name, config_json)

  if not FLAGS.task:
    filename = os.path.join(FLAGS.train_dir, "%s.json" % name)
    with tf.gfile.GFile(filename, "w") as f:
      f.write(config_json)
    tf.logging.info("Wrote %s to %s.", name, filename)


def _log_variable_device_placement():
  """Logs the number of Variable parameters on each device."""
  counter = collections.defaultdict(int)
  for v in tf.global_variables():
    num_params = v.get_shape().num_elements()
    if num_params:
      counter[v.device] += num_params
    else:
      tf.logging.warn("Could not infer num_elements from Variable %s", v.name)

  logstr = ["Device placement:"]
  for device, num_params in counter.iteritems():
    logstr.append(" %s: %.2f M" % (device, float(num_params) / 1e6))
  tf.logging.info("\n".join(logstr))


def main(unused_argv):
  # Create training directory if it doesn't already exist.
  if not tf.gfile.IsDirectory(FLAGS.train_dir):
    tf.logging.info("Creating training directory: %s", FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

  # Set up the model config.
  model_config = configuration.model_config(
      input_file_pattern=FLAGS.input_file_pattern)
  if FLAGS.model_config_overrides:
    model_config.parse_json(FLAGS.model_config_overrides)
  _log_config(model_config, "model_config")

  # Set up the training config.
  training_config = configuration.training_config()
  if FLAGS.training_config_overrides:
    training_config.parse_json(FLAGS.training_config_overrides)
  _log_config(training_config, "training_config")

  tf.logging.info("Building training graph.")
  g = tf.Graph()
  with g.as_default(), g.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
    # Build the model.
    model = skip_thoughts_model.SkipThoughtsModel(
        model_config, mode="train")
    model.build()

    _log_variable_device_placement()

    hooks = [
        # Stop training if loss is NaN.
        tf.train.NanTensorHook(model.total_loss),
        # Log every training step.
        tf.train.LoggingTensorHook(
            {
                "global_step": model.global_step,
                "total_loss": model.total_loss
            },
            every_n_iter=1)
    ]

    # Set up the learning rate and optimizer.
    learning_rate = training.create_learning_rate(training_config,
                                                  model.global_step)
    optimizer = training.create_optimizer(training_config, learning_rate)

    # Set up distributed sync or async training.
    is_chief = (FLAGS.task == 0)
    if FLAGS.sync_replicas:
      optimizer = tf.SyncReplicasOptimizer(
          optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.total_num_replicas)
      hooks.append(optimizer.make_session_run_hook(is_chief))
    else:
      # Startup delay for non-chief asynchronous workers.
      if not is_chief and training_config.startup_delay_steps:
        hooks.append(
            tf.train.GlobalStepWaiterHook(training_config.startup_delay_steps))

    train_tensor = training.create_train_op(training_config, optimizer, model)
    keep_every_n = training_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        max_to_keep=training_config.max_checkpoints_to_keep,
        keep_checkpoint_every_n_hours=keep_every_n,
        save_relative_paths=True)
    scaffold = tf.train.Scaffold(saver=saver)

    # Possibly set a step limit.
    if training_config.number_of_steps:
      hooks.append(
          tf.train.StopAtStepHook(last_step=training_config.number_of_steps))

    # Create the TensorFlow session.
    with tf.train.MonitoredTrainingSession(
        master=FLAGS.master,
        is_chief=is_chief,
        checkpoint_dir=FLAGS.train_dir,
        scaffold=scaffold,
        hooks=hooks,
        save_checkpoint_secs=training_config.save_model_secs,
        save_summaries_steps=None,
        save_summaries_secs=training_config.save_summaries_secs) as sess:

      # Run training.
      while not sess.should_stop():
        sess.run(train_tensor)


if __name__ == "__main__":
  tf.app.run()
