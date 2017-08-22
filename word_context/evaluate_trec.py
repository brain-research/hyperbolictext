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

"""Evaluate the model using per-word perplexity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time

import numpy as np
import tensorflow as tf

from skipthoughts import eval_classification
from skipthoughts import eval_msrp
from skipthoughts import eval_sick
from skipthoughts import eval_trec

from nli import eval_nli
from word_context import word_context_encoder
from word_context import configuration
from word_context import tools

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")

tf.app.flags.DEFINE_string("model_config_overrides", "",
                           "JSON file or JSON string containing configuration "
                           "overrides.")

tf.flags.DEFINE_string("skip_thought_vocab_file", "",
                       "Text file mapping word to word id.")

tf.flags.DEFINE_string("word2vec_embedding_file", "",
                       "File containing serialized embeddings numpy ndarray.")

tf.flags.DEFINE_string("word2vec_vocab_file", "",
                       "Text file mapping word to word id.")

tf.flags.DEFINE_string("eval_tasks", "MultiNLI",
                       "Comma-separated list of the evaluation task of the "
                       "evaluation task. Available tasks: MR, CR, SUBJ, MPQA, "
                       "SICK, MSRP, TREC, MultiNLI.")

tf.flags.DEFINE_string("data_dir", "",
                       "Directory containing the evaluation datasets.")

tf.flags.DEFINE_string("multinli_dir", "",
                       "Directory containing the MultiNLI dataset.")

tf.flags.DEFINE_string("snli_dir", "",
                       "Directory containing the SNLI dataset.")

tf.flags.DEFINE_integer("batch_size", 128, "Batch size for the RNN encoder.")

tf.flags.DEFINE_boolean("use_norm", True,
                        "If True, normalize skip thought vectors to unit L2 "
                        "norm.")

tf.flags.DEFINE_boolean("use_eos", False,
                        "If True, insert the <eos> word during encoding.")

tf.flags.DEFINE_string("nli_eval_method", "logistic",
                       "Classifier for evaluating on NLI datasets."
                       "(logistic / sgd / mlp)")

tf.flags.DEFINE_string("nli_eval_dir", "",
                       "Directory to write NLI evaluation checkpoints"
                       "and summaries.")

tf.logging.set_verbosity(tf.logging.INFO)


def run_task(config, task, checkpoint=None):
  """Evaluates the latest model checkpoint on the given task.

  Args:
    config: Object containing model configuration parameters.
    task: Name of the eval task.
    checkpoint: TF checkpoint to evaluate. If None, the latest checkpoint
      is fetched and evaluated.

  Raises:
    ValueError: If an unrecognized task is passed in --eval_tasks.
  """
  if checkpoint is None:
    skip_thought_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not skip_thought_checkpoint:
      tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                      FLAGS.checkpoint_dir)
      return
  else:
    skip_thought_checkpoint = checkpoint

  # Load the skip thought embeddings and vocabulary.
  skip_thought_emb = tools.load_skip_thought_embeddings(
      skip_thought_checkpoint, config.vocab_size, config.word_embedding_dim)
  _, skip_thought_vocab = tools.load_vocabulary(FLAGS.skip_thought_vocab_file)

  # Load the Word2Vec model.
  word2vec_emb = tools.load_embedding_matrix(FLAGS.word2vec_embedding_file)
  _, word2vec_vocab = tools.load_vocabulary(FLAGS.word2vec_vocab_file)

  # Run vocabulary expansion.
  combined_emb = tools.expand_vocabulary(skip_thought_emb, skip_thought_vocab,
                                         word2vec_emb, word2vec_vocab)

  # Load the encoder.
  g = tf.Graph()
  with g.as_default():
    encoder = word_context_encoder.WordContextEncoder(combined_emb)
    restore_model = encoder.build_graph_from_config(config,
                                                    skip_thought_checkpoint)

  with tf.Session(graph=g) as sess:
    restore_model(sess)
    global_step = tf.train.global_step(sess, "global_step:0")

    class EncoderWrapper(object):
      """Wrapper class for the encode function."""
      @staticmethod
      def encode(data, verbose=False):
        encoded = encoder.encode(
            sess,
            data,
            use_norm=FLAGS.use_norm,
            batch_size=FLAGS.batch_size,
            use_eos=FLAGS.use_eos)
        return np.array(encoded)

    encoder_wrapper = EncoderWrapper()

    tf.logging.info("Running %s evaluation task.", task)
    if task in ["MR", "CR", "SUBJ", "MPQA"]:
      eval_classification.eval_nested_kfold(
          encoder_wrapper, task, FLAGS.data_dir, use_nb=False)
    elif task == "SICK":
      eval_sick.evaluate(encoder_wrapper, evaltest=True, loc=FLAGS.data_dir)
    elif task == "MSRP":
      eval_msrp.evaluate(encoder_wrapper, evalcv=True, evaltest=True,
                         use_feats=True, loc=FLAGS.data_dir)
    elif task == "TREC":
      eval_trec.evaluate(encoder_wrapper, evalcv=True, evaltest=True,
                         loc=FLAGS.data_dir)
    elif task == "MultiNLI":
      results = eval_nli.evaluate(
          encoder_wrapper, FLAGS.multinli_dir,
          os.path.join(FLAGS.nli_eval_dir, "multinli", str(global_step)),
          method=FLAGS.nli_eval_method)
      best_hyperparams = results["best_hyperparameters"]
      print(results[best_hyperparams]["dev"]["overall"])
      print(results[best_hyperparams]["test"]["overall"])
    elif task == "SNLI":
      results = eval_nli.evaluate(
          encoder_wrapper, FLAGS.snli_dir,
          os.path.join(FLAGS.nli_eval_dir, "snli", str(global_step)),
          method=FLAGS.nli_eval_method)
      best_hyperparams = results["best_hyperparameters"]
      print(results[best_hyperparams]["dev"]["overall"])
      print(results[best_hyperparams]["test"]["overall"])
    else:
      raise ValueError("Unrecognized eval_task: %s" % FLAGS.eval_task)

    tf.logging.info("Finished processing evaluation at global step %d.",
                    global_step)


def main(unused_argv):
  if FLAGS.use_eos and FLAGS.hyperbolic:
    raise ValueError("Both use_eos and hyperbolic cannot be true.")

  model_config = configuration.model_config()
  if FLAGS.model_config_overrides:
    model_config.parse_json(FLAGS.model_config_overrides)
  tf.logging.info("model_config: %s",
                  json.dumps(model_config.values(), indent=2))

  for task in FLAGS.eval_tasks.split(","):
    run_task(model_config, task)


if __name__ == "__main__":
  tf.app.run()
