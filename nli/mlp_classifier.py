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

"""Class for training a Multi-Layer Perceptron classifier.

This file contains a class implementing a Multi-Layer Perceptron in TensorFlow
which defines the computation graph and provides methods for training and
evaluating on given data. Please see the class description for usage examples.

We use this implementation instead of tf.contrib.learn.DNNClassifier since to
allow easy implementation of stopping criteria. At the time of writing, the way
to implement stopping criteria in DNNClassifier was through the use of hooks,
however the tensors required to actually compute the validation set performance
were not exposed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import psutil
import tensorflow as tf

EPSILON = 1e-30


def current_memory_gb():
  pid = os.getpid()
  py = psutil.Process(pid)
  memory_use = py.memory_info()[0]/2.**30  # Memory use in GB.
  return memory_use


class MLPClassifier(object):
  """Multi-Layer Perceptron for classification.
  """

  def __init__(self, input_dimension, output_classes, checkpoint_dir,
               depth=3, hidden_size=200, regularization=1e-3,
               nonlinearity='tanh'):
    """Setup TensorFlow graph for training MLP.

    Args:
      input_dimension: Size of input features.
      output_classes: Number of target classes.
      checkpoint_dir: Directory to save models and summaries.
      depth: (Optional) number of hidden layers.
      hidden_size: (Optional) size of each hidden layer.
      regularization: (Optional) tuning parameter for L2-regularization.
      nonlinearity: (Optional) type of nonlinearity for hidden layers. Supports
        only 'none' and 'tanh' for now.

    Raises:
      ValueError: If an unsupported nonlinearity is provided.
    """
    if nonlinearity not in ['none', 'tanh']:
      raise ValueError('Unrecognized nonlinearity: %s' % nonlinearity)

    self.input_dimension = input_dimension
    self.output_classes = output_classes
    self.checkpoint_dir = checkpoint_dir

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.batch_size = tf.placeholder(tf.int64, shape=[])
      self.filename = tf.placeholder(tf.string, shape=[])
      training_dataset = self._build_input_pipeline(self.batch_size,
                                                    self.filename)
      validation_dataset = self._build_input_pipeline(self.batch_size,
                                                      self.filename,
                                                      shuffle=False)

      iterator = tf.contrib.data.Iterator.from_structure(
          training_dataset.output_types,
          training_dataset.output_shapes)
      self.input, self.target = iterator.get_next()

      self.training_init_op = iterator.make_initializer(training_dataset)
      self.validation_init_op = iterator.make_initializer(validation_dataset)

      self.output = self._build_model(depth, hidden_size, nonlinearity)
      penalty = sum([tf.nn.l2_loss(weight)
                     for weight in tf.trainable_variables()])
      one_hot_labels = tf.one_hot(self.target, output_classes,
                                  dtype=tf.float32)
      self.loss = - tf.reduce_mean(tf.reduce_sum(
          one_hot_labels * tf.log(self.output + EPSILON), axis=1))
      self.accuracy = tf.reduce_mean(tf.cast(
          tf.equal(tf.cast(tf.argmax(self.output, axis=1), tf.int32),
                   self.target),
          tf.float32))
      tf.summary.scalar('mlp_loss', self.loss)
      tf.summary.scalar('mlp_accuracy', self.accuracy)

      self.learning_rate = tf.placeholder(tf.float32)
      adadelta = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
      self.train_op = adadelta.minimize(self.loss + regularization * penalty)

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.session = tf.Session(config=config)
      self.saver = tf.train.Saver()

      self.merged_summaries = tf.summary.merge_all()

  def close_session(self):
    """Close the TensorFlow session for this object."""
    self.session.close()

  def get_output_placeholder(self):
    """Return the output placeholder.

    Returns:
      self.output: TensorFlow placeholder for the output of this model.
        Useful when this module is used as part of a bigger model.
    """
    return self.output

  def fit(self, training_filename, dev_filename, batch_size=32, max_epochs=100,
          initial_learning_rate=1e-1, logging_frequency=1000):
    """Trains the model on input data.

    Training is performed for maximum number of epochs or until performance
    on validation set stops improving. Dev features are used to track model
    performance.

    Args:
      training_filename: File containing training TFRecords.
      dev_filename: File containing validation TFRecords.
      batch_size: Batch size.
      max_epochs: Maximum number of epochs to run for.
      initial_learning_rate: (Optional) scalar float for learning rate.
      logging_frequency: (Optional) number of updates after which to record
        training loss and accuracy.

    Returns:
      updates: The number of updates performed during training.
    """
    tf.logging.info('Training model ...')
    tf.logging.info('Initial Learning Rate = %.5e', initial_learning_rate)
    learning_rate = initial_learning_rate
    with self.graph.as_default():
      self.session.run(tf.local_variables_initializer())
      self.session.run(tf.global_variables_initializer())
      train_writer = tf.summary.FileWriter(
          os.path.join(self.checkpoint_dir, 'train'))
      dev_writer = tf.summary.FileWriter(
          os.path.join(self.checkpoint_dir, 'dev'))
      updates = 0
      self._save(updates)
      max_dev_accuracy, min_dev_loss = 0., 1e12
      stopping_criterion = False

      for epoch in range(max_epochs):
        if stopping_criterion: break
        self.session.run(self.training_init_op,
                         feed_dict={self.filename: training_filename,
                                    self.batch_size: batch_size})
        epoch_finished = False
        while not epoch_finished:
          try:
            summary, loss, accuracy, _ = self.session.run(
                [self.merged_summaries,
                 self.loss,
                 self.accuracy,
                 self.train_op],
                feed_dict={self.learning_rate: learning_rate})
            updates += 1

            if updates % logging_frequency == 0:
              tf.logging.info('Update %d Training Loss = %.4f '
                              'Accuracy = %.4f Learning Rate = %.2e',
                              updates, loss, accuracy, learning_rate)
              train_writer.add_summary(summary, updates)
              # Log memory usage
              summary = tf.Summary()
              value = summary.value.add()
              value.simple_value = current_memory_gb()
              value.tag = 'memory_usage_gb'
              train_writer.add_summary(summary, updates)
              train_writer.flush()

          except tf.errors.OutOfRangeError:
            epoch_finished = True
            tf.logging.info('Finished Epoch %d', epoch)

            dev_loss, dev_accuracy, _ = self.predict(dev_filename,
                                                     reload_checkpoint=False)
            tf.logging.info('Update %d Dev Loss = %.4f '
                            'Accuracy = %.4f Learning Rate = %.2e',
                            updates, dev_loss, dev_accuracy, learning_rate)
            # Log loss and accuracy to summary writer
            summary = tf.Summary()
            value = summary.value.add()
            value.simple_value = dev_loss
            value.tag = 'mlp_loss'
            dev_writer.add_summary(summary, updates)
            summary = tf.Summary()
            value = summary.value.add()
            value.simple_value = dev_accuracy
            value.tag = 'mlp_accuracy'
            dev_writer.add_summary(summary, updates)
            dev_writer.flush()

            if dev_accuracy > max_dev_accuracy:
              max_dev_accuracy = dev_accuracy
              self._save(updates)

            if dev_loss < min_dev_loss:
              min_dev_loss = dev_loss

            # Stopping criterion
            if (dev_loss - min_dev_loss) / min_dev_loss > 0.5:
              tf.logging.info('Done training -- stopping criterion met.')
              stopping_criterion = True

      train_writer.close()
      dev_writer.close()
      tf.logging.info('Training complete.')

    return updates

  def predict(self, filename, batch_size=512, reload_checkpoint=True):
    """Run input data through the model.

    Args:
      filename: File containing TFRecord data.
      batch_size: Number of samples to evaluate at once.
      reload_checkpoint: (Optional) to reload the latest checkpoint or not.

    Returns:
      loss: Average loss across all samples.
      accuracy: Average accuracy.
      outputs: 2d numpy array holding Softmax outputs for each sample.
        Size of the array is (features.shape[0], self.output_classes).
    """
    self.session.run(self.validation_init_op,
                     feed_dict={self.filename: filename,
                                self.batch_size: batch_size})
    total_loss, total_correct, total = 0., 0., 0
    all_outputs = []
    if reload_checkpoint: self._load()
    epoch_finished = False
    while not epoch_finished:
      try:
        loss, accuracy, outputs = self.session.run(
            [self.loss, self.accuracy, self.output])
        total += outputs.shape[0]
        total_loss += loss * outputs.shape[0]
        total_correct += accuracy * outputs.shape[0]
        all_outputs.append(outputs)
      except tf.errors.OutOfRangeError:
        epoch_finished = True
    return (total_loss / total, total_correct / total, np.vstack(all_outputs))

  def _build_input_pipeline(self, batch_size, filename, shuffle=True):
    """Prefetches values from disk into an input queue."""
    def _parse_function(example_proto_batch):
      features = {
          'features': tf.FixedLenFeature([self.input_dimension],
                                         dtype=tf.float32),
          'label': tf.FixedLenFeature([], dtype=tf.int64),
      }
      deserialized_batch = tf.parse_example(example_proto_batch, features)
      return deserialized_batch['features'], tf.cast(
          deserialized_batch['label'], tf.int32)

    dataset = tf.contrib.data.TFRecordDataset(filename)
    if shuffle: dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_function)
    return dataset

  def _build_model(self, depth, hidden_size, nonlinearity):
    """Constructs the Multi-Layer Perceptron.

    Args:
      depth: Number of hidden layers.
      hidden_size: Size of each hidden layer.
      nonlinearity: Tpe of nonlinearity for hidden layers. Supports
        only 'tanh' for now.

    Returns:
      output: TensorFlow placeholder for the output of the network.
    """
    def _glorot(d1, d2):
      return np.sqrt(6./(d1+d2))

    # Hidden layers
    activations = self.input
    for i in range(depth):
      with tf.variable_scope('layer_%d' % i):
        input_size = hidden_size if i > 0 else self.input_dimension
        weight = tf.Variable(
            tf.random_normal([input_size, hidden_size],
                             stddev=_glorot(input_size, hidden_size)),
            name='W_%d' % i, dtype=tf.float32)
        bias = tf.Variable(tf.zeros([hidden_size]), name='b_%d' % i,
                           dtype=tf.float32)
        activations = tf.matmul(activations, weight) + bias
        if nonlinearity == 'tanh':
          activations = tf.tanh(activations)

    # Output layer
    with tf.variable_scope('layer_output'):
      weight = tf.Variable(
          tf.random_normal([hidden_size, self.output_classes],
                           stddev=_glorot(hidden_size, self.output_classes)),
          name='W_out', dtype=tf.float32)
      bias = tf.Variable(tf.zeros([self.output_classes]), name='b_out',
                         dtype=tf.float32)
      output = tf.nn.softmax(tf.matmul(activations, weight) + bias)

    return output

  def _save(self, step):
    """Save current model parameters.

    Args:
      step: Integer step for checkpointing the model.
    """
    tf.logging.info('Saving checkpoint for step %d ...', step)
    self.saver.save(self.session,
                    os.path.join(self.checkpoint_dir, 'model.ckpt'),
                    global_step=step)

  def _load(self):
    """Load latest checkpoint."""
    ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
    tf.logging.info('Loading latest checkpoint from %s ...', ckpt)
    self.saver.restore(self.session, ckpt)
