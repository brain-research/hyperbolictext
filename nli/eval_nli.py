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

"""Functions to evaluate pre-trained vectors on the NLI task.

This file holds methods which use a pre-trained TensorFlow model to encode
sentences in the Multi-NLI corpus. Then it trains a Logistic Regression model or
an MLP classifier to predict the three classes in the dataset -- entail,
contradict, neutral.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
from collections import namedtuple
import logging
import operator
import os

import numpy as np
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from nli import mlp_classifier
from nli import tfrecord_creator

# NLIData is a tuple consisting four lists:
#   sents_pre: Premise sentences.
#   sents_hyp: Hypothesis sentences.
#   labels: Gold label for relation between corresponding premise and
#     hypothesis sentences. One of 'contradiction', 'entailment' or
#     'neutral'.
#   annotations: Annotations of the type of data. None if not available
NLIData = namedtuple('NLIData',
                     ['sents_pre', 'sents_hyp', 'labels', 'annotations'])


def current_memory_gb():
  pid = os.getpid()
  py = psutil.Process(pid)
  memory_use = py.memory_info()[0]/2.**30  # Memory use in GB.
  return memory_use


def load_data(path_prefix):
  """Load NLI data from given location.

  Args:
    path_prefix: Location+file-prefix containing NLI data (<string>).

  Returns:
    train_data: NLIData for the training set.
    dev_data: NLIData for the development set.
    test_data: NLIData for the test set.
  """

  train_data = NLIData([], [], [], [])
  dev_data = NLIData([], [], [], [])
  test_data = NLIData([], [], [], [])

  def read_file(suffix, nli_tuple):
    with open('%s_%s.jsonl'%(path_prefix, suffix)) as f:
      for line in f:
        data = ast.literal_eval(line.rstrip())
        if data['gold_label'] == '-': continue
        nli_tuple.sents_pre.append(data['sentence1'])
        nli_tuple.sents_hyp.append(data['sentence2'])
        nli_tuple.labels.append(data['gold_label'])
        if 'genre' in data:
          nli_tuple.annotations.append(data['genre'])
        else:
          nli_tuple.annotations.append(None)

  read_file('train', train_data)
  read_file('dev', dev_data)
  read_file('test', test_data)

  return train_data, dev_data, test_data


def _encode(nli_data, encoder, dim, interactions, dtype, batch_size=2048):
  """Encode premises and hypotheses in MultiNLIData using the provided encoder.

  Args:
    nli_data: NLIData object.
    encoder: Object implementing the 'encode' method which returns a vector
      for each input sentence.
    dim: Dimension of the encoded vectors.
    interactions: Boolean. Let u, v be the vectors for the premise and
      hypothesis sentences, then the returned vector is [ u || v ] if this is
      set to False, otherwise [ u || v || u * v || u - v ].
    dtype: 'float32' or 'float64'.
    batch_size: Number of sentences to encode in one iteration. These are
      copied so higher number will lead to increased memory usage.

  Returns:
    vectors: Numpy array of size (len(lni_data.sents_pre), 2*dim).
  """
  if interactions:
    vectors = np.zeros((len(nli_data.sents_pre), 4*dim), dtype=dtype)
  else:
    vectors = np.zeros((len(nli_data.sents_pre), 2*dim), dtype=dtype)

  batch_indices = np.arange(0, len(nli_data.sents_pre), batch_size)
  for start_index in batch_indices:
    end_index = min(len(nli_data.sents_pre), start_index + batch_size)

    vectors[start_index:end_index, :dim] = np.nan_to_num(np.vstack(
        encoder.encode(nli_data.sents_pre[start_index:end_index])))
    vectors[start_index:end_index, dim:2*dim] = np.nan_to_num(np.vstack(
        encoder.encode(nli_data.sents_hyp[start_index:end_index])))

    if interactions:
      vectors[start_index:end_index, 2*dim:3*dim] = (
          vectors[start_index:end_index, :dim] *
          vectors[start_index:end_index, dim:2*dim])
      vectors[start_index:end_index, 3*dim:] = (
          vectors[start_index:end_index, :dim] -
          vectors[start_index:end_index, dim:2*dim])

  return vectors


def _get_encoder_dimension(encoder):
  """Find dimensionality of encoded vectors.

  Args:
    encoder: Object implementing the encode() method which takes
      a list of strings as input and returns a list of
      numpy vectors as output.

  Returns:
    dimension: Integer size of the encoded vectors.
  """
  vector = encoder.encode(['test sentence'])
  dimension = vector[0].shape[0]
  return dimension


def evaluate(encoder, path_prefix, eval_dir,
             interactions=True, method='logistic'):
  """Evaluate the encoder using 3-way classification.

  Features are stored in 'float64' format if method == 'logistic'. Note that
  this means with logistic method the maximum size of the training dataset is
  limited to approximately 350K samples.

  Args:
    encoder: Object implementing the encode() method which takes
      a list of strings as input and returns a list of
      numpy vectors as output.
    path_prefix: Location+file-prefix of NLI dataset (<string>).
      We expect 3 files at this location -
      <path_prefix>_train.jsonl,
      <path_prefix>_dev.jsonl,
      <path_prefix>_test.jsonl.
    eval_dir: Directory to store summaries and checkpoints. Only used when
      method == 'mlp'.
    interactions: Boolean. Let u, v be the vectors for the premise and
      hypothesis sentences, then the returned vector is [ u || v ] if this is
      set to False, otherwise [ u || v || u * v || u - v ].
    method: Classifier training method (logistic / mlp).

  Returns:
    results: A dictionary mapping hyperparameter values to genre-wise dev and
      test accuracies, as well as the hyperparameter configuration for the best
      overall performance on the dev set.

  Raises:
    ValueError: If the classifier training method is unrecognized.
  """
  if method == 'logistic':
    dtype = 'float64'  # scikit-learn logistic regression only supports float64
  else:
    dtype = 'float32'  # saves memory

  logging.info('Loading data ...')
  train, dev, test = load_data(path_prefix)
  logging.info('%d training examples', len(train.sents_pre))
  logging.info('%d dev examples', len(dev.sents_pre))
  logging.info('%d test examples', len(test.sents_pre))

  dim = _get_encoder_dimension(encoder)
  logging.info('Extracting train features ...')
  train_features = _encode(train, encoder, dim, interactions, dtype)
  logging.info('Extracting dev features ...')
  dev_features = _encode(dev, encoder, dim, interactions, dtype)
  logging.info('Extracting test features ...')
  test_features = _encode(test, encoder, dim, interactions, dtype)
  logging.info('Converting labels ...')
  le = LabelEncoder()
  le.fit(train.labels)
  train_label = le.transform(train.labels)
  dev_label = le.transform(dev.labels)
  test_label = le.transform(test.labels)
  logging.info('Feature extraction done.')

  logging.info('Evaluating ...')
  if method == 'logistic':
    results = _evaluate_logistic(train_features, train_label,
                                 dev_features, dev_label, dev.annotations,
                                 test_features, test_label, test.annotations)
  elif method == 'mlp':
    # Write to disk and free up memory
    def _to_tfrecord(features, label, name):
      tfrecord_filename = os.path.join(eval_dir, name + '.tfrecords')
      logging.info('Saving TFRecords to %s.', tfrecord_filename)
      tfrecord_creator.convert_to_tfrecord(features, label,
                                           tfrecord_filename)
      return tfrecord_filename

    logging.info('Converting training dataset to TFRecords.')
    logging.info('Memory used before: %.3fGB', current_memory_gb())
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)
    train_file = _to_tfrecord(train_features, train_label, 'train')
    del train_features, train_label
    dev_file = _to_tfrecord(dev_features, dev_label, 'dev')
    del dev_features
    test_file = _to_tfrecord(test_features, test_label, 'test')
    del test_features
    logging.info('Memory used after: %.3fGB', current_memory_gb())
    logging.info('Saved TFRecords.')

    input_dimension = 4*dim if interactions else 2*dim
    results = _evaluate_mlp(train_file, dev_file, test_file,
                            dev_label, test_label,
                            dev.annotations, test.annotations,
                            eval_dir, input_dimension)
  else:
    raise ValueError('Unrecognised training method.')

  overall_accuracies = [(k, v['dev']['overall'])
                        for k, v in results.iteritems()]
  best = max(overall_accuracies, key=operator.itemgetter(1))
  best_hyperparams = best[0]
  logging.info('Best Dev Accuracy = %.4f (Hyperparams = %s)',
               best[1], str(best_hyperparams))
  results['best_hyperparameters'] = best_hyperparams

  return results


def _compile_results(ground, predicted, annotation):
  """Compute genre-wise and overall accuracies.

  Args:
    ground: Ground-truth labels (numpy array).
    predicted: Predicted labels (numpy array).
    annotation: Annotations (list of strings).

  Returns:
    results: Dict with keys for each non-'None' genre in annotation, and one
      key for 'overall'. Values are accuracies.
  """
  correct = {'overall': 0}
  count = {'overall': 0}
  for ii in range(ground.shape[0]):
    if annotation[ii] and annotation[ii] not in correct:
      correct[annotation[ii]] = 0
      count[annotation[ii]] = 0
    if annotation[ii]: count[annotation[ii]] += 1
    count['overall'] += 1
    if ground[ii] == predicted[ii]:
      if annotation[ii]: correct[annotation[ii]] += 1
      correct['overall'] += 1

  results = {}
  for k in correct:
    results[k] = float(correct[k]) / count[k]

  return results


def _evaluate_mlp(train_file, dev_file, test_file,
                  dev_label, test_label,
                  dev_annotation, test_annotation,
                  eval_dir, dimension):
  """Run Multi-Layer Perceptron Classifier.

  Args:
    train_file: File containing TFRecords for training.
    dev_file: File containing TFRecords for validation.
    test_file: File containing TFRecords for testing.
    dev_label: Numpy array of size (N,) containing labels.
    test_label: Numpe array of size (N,) containing labels
    dev_annotation: List of size (N,) containing strings.
    test_annotation: List of size (N,) containing strings.
    eval_dir: Directory to save checkpoints and summaries.
    dimension: Feature dimension.

  Returns:
    results: A dictionary mapping hyperparameter values to results on the dev
      and test sets.
  """
  logging.info('MLP Classifier ...')
  scan_l2reg = [1e-5, 1e-3]
  scan_lr = [1e-2, 1e-1]
  results = {}
  for l2reg in scan_l2reg:
    for lr in scan_lr:
      logging.info('Running for C = %.4e LR = %.3f ...', l2reg, lr)
      hyperparameters = (('C', l2reg), ('LR', lr))
      results[hyperparameters] = {}
      # Make checkpoint directory
      checkpoint_dir = os.path.join(eval_dir, 'c_%.2e_lr_%.3f' % (l2reg, lr))
      logging.info('Storing checkpoints in %s', checkpoint_dir)
      if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
      # Run training
      clf = mlp_classifier.MLPClassifier(dimension, 3, checkpoint_dir,
                                         regularization=l2reg)
      clf.fit(train_file,
              dev_file,
              initial_learning_rate=lr)
      # Evaluate on dev set
      _, _, dev_out = clf.predict(dev_file)
      yhat = np.argmax(dev_out, axis=1)
      results[hyperparameters]['dev'] = _compile_results(
          dev_label, yhat, dev_annotation)
      # Evaluate on test set
      _, _, test_out = clf.predict(test_file)
      yhat = np.argmax(test_out, axis=1)
      results[hyperparameters]['test'] = _compile_results(
          test_label, yhat, test_annotation)
      logging.info('C=%.2e LR = %.3f Dev Accuracy=%.4f Test Accuracy=%.4f',
                   l2reg, lr, results[hyperparameters]['dev']['overall'],
                   results[hyperparameters]['test']['overall'])
      # Delete the session and classifier object
      logging.info('Closing session.')
      clf.close_session()
      del clf

  return results


def _evaluate_logistic(train_features, train_label,
                       dev_features, dev_label, dev_annotation,
                       test_features, test_label, test_annotation):
  """Run Logistic Regression Classifier.

  Args:
    train_features: Numpy matrix of size (N, D).
    train_label: Numpy array of size (N,).
    dev_features: Numpy matrix of size (N, D).
    dev_label: Numpy array of size (N,).
    dev_annotation: List of size (N,) containing strings.
    test_features: Numpy matrix of size (N, D).
    test_label: Numpy array of size (N,).
    test_annotation: List of size (N,) containing strings.

  Returns:
    results: A dictionary mapping hyperparameter values to results on the dev
      and test sets.
  """
  logging.info('Logistic Regression ...')
  scan_l2reg = [2**t for t in range(-5, 5, 2)]
  results = {}
  for l2reg in scan_l2reg:
    logging.info('Running for C = %.4f ...', l2reg)
    hyperparameters = ('C', l2reg)
    results[hyperparameters] = {}
    # Run training
    clf = LogisticRegression(C=l2reg, solver='sag', verbose=False)
    clf.fit(train_features, train_label)
    # Evaluate on dev set
    yhat = clf.predict(dev_features)
    results[hyperparameters]['dev'] = _compile_results(
        dev_label, yhat, dev_annotation)
    # Evaluate on test set
    yhat = clf.predict(test_features)
    results[hyperparameters]['test'] = _compile_results(
        test_label, yhat, test_annotation)
    logging.info('C=%.4f Dev Accuracy=%.4f Test Accuracy=%.4f',
                 l2reg, results[hyperparameters]['dev']['overall'],
                 results[hyperparameters]['test']['overall'])

  return results
