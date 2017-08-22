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

"""Compute an expanded vocabulary of embeddings using a word2vec model.

This script loads the word embeddings from a trained skip thought model and from
a trained word2vec model (typically with a larger vocabulary). It trains a
linear regression model without regularization to learn a linear mapping from
the word2vec embedding space to the skip thought embedding space. The model is
then applied to all words in the word2vec vocabulary, yielding "skip thought
embeddings" for the union of the two vocabularies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import numpy as np
import sklearn.linear_model
import tensorflow as tf


def load_skip_thought_embeddings(checkpoint_path, vocab_size, embedding_dim):
  """Loads the embedding matrix from a skip thought model checkpoint.

  Args:
    checkpoint_path: Model checkpoint file or directory containing a checkpoint
        file.
    vocab_size: Number of words in the vocabulary.
    embedding_dim: Word embedding dimension.

  Returns:
    word_embedding: A numpy array of shape [vocab_size, embedding_dim].

  Raises:
    ValueError: If no checkpoint file matches checkpoint_path.
  """
  if tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    if not checkpoint_file:
      raise ValueError("No checkpoint file found in %s" % checkpoint_path)
  else:
    checkpoint_file = checkpoint_path

  g = tf.Graph()
  with g.as_default():
    word_embedding_tensor = tf.get_variable(
        name="word_embedding", shape=[vocab_size, embedding_dim])

    saver = tf.train.Saver()

  with tf.Session(graph=g) as sess:
    tf.logging.info("Loading skip thought embedding matrix from %s",
                    checkpoint_file)
    saver.restore(sess, checkpoint_file)
    word_embedding = sess.run(word_embedding_tensor)
    tf.logging.info("Loaded skip thought embedding matrix of shape %s",
                    word_embedding.shape)

  return word_embedding


def expand_vocabulary(skip_thought_emb, skip_thought_vocab, word2vec_emb,
                      word2vec_vocab):
  """Runs vocabulary expansion on a skip thought model using a word2vec model.

  This function trains a linear regression model without regularization to learn
  a linear mapping from the word2vec embedding space to the skip thought
  embedding space. The model is then applied to all words in the word2vec
  vocabulary, yielding "skip thought embeddings" for the union of the two
  vocabularies.

  Args:
    skip_thought_emb: A numpy array of shape [skip_thought_vocab_size,
        skip_thought_embedding_dim].
    skip_thought_vocab: A dictionary of word to id.
    word2vec_emb: A numpy array of shape [word2vec_vocab_size,
        word2vec_embedding_dim].
    word2vec_vocab: A dictionary of word to id.

  Returns:
    combined_emb: A dictionary mapping words to embedding vectors.
  """
  # Find words shared between the two vocabularies.
  tf.logging.info("Finding shared words")
  shared_words = [w for w in word2vec_vocab if w in skip_thought_vocab]

  # Select embedding vectors for shared words.
  tf.logging.info("Selecting embeddings for %d shared words", len(shared_words))
  shared_st_emb = skip_thought_emb[
      [skip_thought_vocab[w] for w in shared_words]]
  shared_w2v_emb = word2vec_emb[[word2vec_vocab[w] for w in shared_words]]

  # Train a linear regression model on the shared embedding vectors.
  tf.logging.info("Training linear regression model")
  model = sklearn.linear_model.LinearRegression()
  model.fit(shared_w2v_emb, shared_st_emb)

  # Create the expanded vocabulary.
  tf.logging.info("Creating embeddings for expanded vocabuary")
  combined_emb = collections.OrderedDict()
  for w in word2vec_vocab:
    # Ignore words with underscores (spaces).
    if "_" not in w:
      w_emb = model.predict(word2vec_emb[word2vec_vocab[w]].reshape(1, -1))
      combined_emb[w] = w_emb.reshape(-1)

  for w in skip_thought_vocab:
    combined_emb[w] = skip_thought_emb[skip_thought_vocab[w]]

  tf.logging.info("Created expanded vocabulary of %d words", len(combined_emb))

  return combined_emb


def save_embedding_map(embeddings, output_dir):
  """Saves a word embedding map.

  Args:
    embeddings: A dictionary mapping words to embedding vectors.
    output_dir: Directory in which to save the dictionary and embedding matrix.
  """
  dictionary = embeddings.keys()
  embeddings = np.array(embeddings.values())

  # Write the dictionary.
  dictionary_file = os.path.join(output_dir, "dictionary.txt")
  with tf.gfile.GFile(dictionary_file, "w") as f:
    f.write("\n".join(dictionary))
  tf.logging.info("Wrote dictionary file to %s", dictionary_file)

  # Write the embeddings.
  embeddings_file = os.path.join(output_dir, "embeddings.npy")
  np.save(embeddings_file, embeddings)
  tf.logging.info("Wrote embeddings file to %s", embeddings_file)


def load_vocabulary(filename):
  """Loads a vocabulary file.

  Args:
    filename: Path to text file containing newline separated words.

  Returns:
    reverse_vocab: A list mapping word id to word.
    vocab: A dictionary mapping word to word id.
  """
  tf.logging.info("Reading vocabulary from %s", filename)
  with tf.gfile.GFile(filename, mode="r") as f:
    lines = list(f.readlines())
  reverse_vocab = [line.decode("utf-8").strip() for line in lines]
  tf.logging.info("Read vocabulary of size %d", len(reverse_vocab))
  vocab = collections.OrderedDict([(w, i) for i, w in enumerate(reverse_vocab)])
  return reverse_vocab, vocab


def load_embedding_matrix(filename):
  """Loads an embedding matrix.

  Args:
    filename: Path to serialized numpy ndarray of shape
        [num_words, embedding_dim].

  Returns:
    A numpy ndarray of shape [num_words, embedding_dim].
  """
  tf.logging.info("Loading embedding matrix from %s", filename)
  with open(filename, "r") as f:
    # Note: tf.gfile.GFile doesn't work here because np.load() expects to be
    # able to call f.seek() with 3 arguments.
    embedding_matrix = np.load(f)

  tf.logging.info("Loaded embedding matrix of shape %s", embedding_matrix.shape)
  return embedding_matrix


def create_embedding_map(reverse_vocab, embedding_matrix):
  """Returns a dictionary mapping word to word embedding."""
  return collections.OrderedDict(zip(reverse_vocab, embedding_matrix))


if __name__ == "__main__":
  tf.app.run()
