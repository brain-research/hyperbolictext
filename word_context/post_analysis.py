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

"""Analyse trained word context models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

from word_context import configuration
from word_context import word_context_model
from word_context import tools

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", None,
                       "Path to checkpoint or the directory containing the "
                       "checkpoint to analyse.")

tf.flags.DEFINE_string("vocab_file", "",
                       "Text file mapping word to id.")

tf.flags.DEFINE_string("output_dir", "",
                       "Directory to store the output analysis to.")

tf.flags.DEFINE_string("sentences_file", "",
                       "Path to txt file containing sample sentences to "
                       "encode and visualize.")

tf.flags.DEFINE_boolean("plot", True,
                        "Whether to plot embeddings or not.")

tf.flags.DEFINE_string("mr_data_dir", "",
                       "Path to the Movie Review dataset. This should contain "
                       "the files rt-polarity.pos and rt-polarity.neg.")

# List of types to store output probabilities for.
TARGET_TYPES = ["the", "you", "alejo", "noon", "pablo", "gravel", "she",
                "decrepit", "conference"]

# List of tokens to compute the nearest neighbours to in embedding space.
SEED_TOKENS = ["<unk>", "the", "james", "least", "car", "red", "how",
               "organization"]

# List of test sentences for which we will compute norms.
TEST_SENTENCES = [
    "james",
    "car",
    "red car",
    "driving a red car on the road",
    "james was driving a red car on the road",
    "brain",
    "google brain",
    "bhuwan is doing an internship",
    "bhuwan is doing an internship at google brain",
    "sky",
    "blue sky",
    "beautiful",
    "such a beautiful",
    "such a beautiful blue sky today",
]


def _sq_euclidean_distance(x, y):
  """Compute squared euclidean distance between batches x and y.

  Args:
    x: 2D Numpy array.
    y: 2D Numpy array.

  Returns:
    distances: 2D Numpy array such that distances[i,j] =
      squared_L2_distance(x[i,:], y[j,:]).
  """
  sq_norm_x = (x ** 2).sum(axis=1)[:,None]
  sq_norm_y = (y ** 2).sum(axis=1)[None,:]
  inner_product = np.dot(x, np.transpose(y))
  distances = sq_norm_x + sq_norm_y - 2 * inner_product
  return np.clip(distances, 0., np.inf)


def _hyperbolic_distance(x, y):
  """Compute hyperbolic distance between batches x and y.

  Args:
    x: 2D Numpy array.
    y: 2D Numpy array.

  Returns:
    distances: 2D Numpy array such that distances[i,j] =
      hyperbolic(x[i,:], y[j,:]).
  """
  sq_norm_x = (x ** 2).sum(axis=1)[:,None]
  sq_norm_y = (y ** 2).sum(axis=1)[None,:]
  inner_product = np.dot(x, np.transpose(y))
  sq_norm_xy = sq_norm_x + sq_norm_y - 2 * inner_product
  return np.arccosh(1. + np.clip(
      2. * sq_norm_xy / np.clip((1. - sq_norm_x) * (1. - sq_norm_y), 1e-15, 1.),
      0., np.inf))


def _cosine_similarity(x, y):
  """Compute cosine similarity between batches x and y.

  Args:
    x: 2D Numpy array.
    y: 2D Numpy array.

  Returns:
    similarities: 2D Numpy array such that similarities[i, j] =
    cosine(x[i,:], y[j,:]).
  """
  normalized_x = x / np.linalg.norm(x, axis=1, keepdims=True)
  normalized_y = y / np.linalg.norm(y, axis=1, keepdims=True)

  return np.dot(normalized_x, np.transpose(normalized_y))


def _create_restore_fn(checkpoint_path, saver):
  """Creates a function that restores a model from checkpoint.

  Args:
    checkpoint_path: Checkpoint file or a directory containing a checkpoint
      file.
    saver: Saver for restoring variables from the checkpoint file.

  Returns:
    restore_fn: A function such that restore_fn(sess) loads model variables
      from the checkpoint file.
    checkpoint_path: Path to the model checkpoint that will be loaded

  Raises:
    ValueError: If checkpoint_path does not refer to a checkpoint file or a
      directory containing a checkpoint file.
  """
  if tf.gfile.IsDirectory(checkpoint_path):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if not latest_checkpoint:
      raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
    checkpoint_path = latest_checkpoint

  def _restore_fn(sess):
    tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Successfully loaded checkpoint: %s",
                    os.path.basename(checkpoint_path))

  return _restore_fn, checkpoint_path


def save_euclidean_norms(output_embeddings, vocab):
  """Compute euclidean norms and save in sorted order.

  Args:
    output_embeddings: Numpy array containing embeddings, one per row.
    vocab: List of tokens in the same order as they appear in output_embeddings.
  """
  norms = np.linalg.norm(output_embeddings, axis=1)

  sorted_idx = np.argsort(norms)
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "sorted_euclidean_norms.txt"),
                    "w")
  f.write("\n".join(["%s\t%.3f" % (vocab[ii], norms[ii])
                     for ii in sorted_idx]))
  f.close()


def save_hyperbolic_norms(output_embeddings, vocab):
  """Compute hyperbolic norms and save in sorted order.

  Args:
    output_embeddings: Numpy array containing embeddings, one per row.
    vocab: List of tokens in the same order as they appear in output_embeddings.
  """
  hyp_norms = np.squeeze(
      _hyperbolic_distance(output_embeddings,
                           np.zeros((1, output_embeddings.shape[1]))))

  sorted_idx = np.argsort(hyp_norms)
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "sorted_hyperbolic_norms.txt"),
                    "w")
  f.write("\n".join(["%s\t%.3f" % (vocab[ii], hyp_norms[ii])
                     for ii in sorted_idx]))
  f.close()


def save_nearest_neighbours(output_embeddings, vocab, seeds, num_nearest, metric):
  """Find nearest neighbours in embedding space to given seeds.

  Args:
    output_embeddings: Numpy array containing embeddings, one per row.
    vocab: List of tokens in the same order as they appear in output_embeddings.
    seeds: List of tokens to compute the nearest neighbours to.
    num_nearest: Number of neighbours to output.
    metric: "hyperbolic" or "cosine" or None.
  """
  seed_emb = np.zeros((len(seeds), output_embeddings.shape[1]))
  for ix, token in enumerate(vocab):
    if token in seeds:
      seed_emb[seeds.index(token), :] = output_embeddings[ix,:]

  if metric == "hyperbolic":
    distances = _hyperbolic_distance(seed_emb, output_embeddings)
  elif metric == "cosine":
    distances = - _cosine_similarity(seed_emb, output_embeddings)
  else:
    distances = _sq_euclidean_distance(seed_emb, output_embeddings)
  sorted_index = np.argsort(distances, axis=1)

  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "nearest_neighbours.txt"),
                    "w")
  for ix in range(sorted_index.shape[0]):
    f.write(seeds[ix] + " (%.3f)" % distances[ix, vocab.index(seeds[ix])] +
            " " + " ". join(["%s (%.3f)" % (vocab[ii], distances[ix,ii])
                             for ii in sorted_index[ix, :num_nearest]]) +
            "\n")
  f.close()


def build_hierarchy(thought_vectors, sentences, metric):
  """Recursively find the nearest neighbour with greater norm.

  Args:
    thought_vectors: 2D numpy array containing the thought vectors.
    sentences: Corresponding string sentences.
    metric: To compute distances.
  """
  if metric == "hyperbolic":
    distances = _hyperbolic_distance(thought_vectors, thought_vectors)
  elif metric == "cosine":
    distances = - _cosine_similarity(thought_vectors, thought_vectors)
  else:
    distances = _sq_euclidean_distance(thought_vectors, thought_vectors)

  sorted_index = np.argsort(distances, axis=1)

  hyp_norms = np.squeeze(
      _hyperbolic_distance(thought_vectors,
                           np.zeros((1, thought_vectors.shape[1]))))

  def _find_closest(n):
    """Return closest neighbour of n-th sentence with greater norm."""
    current_norm = hyp_norms[n]
    neighbour_norms = hyp_norms[sorted_index[n, 1:]]
    ii = 0
    while True:
      if ii == neighbour_norms.shape[0]: break
      if neighbour_norms[ii] > current_norm:
        break
      ii += 1
    if ii == neighbour_norms.shape[0]:
      return None
    else:
      return sorted_index[n, 1 + ii]

  f = tf.gfile.Open(os.path.join(FLAGS.output_dir,
                                 "sentence_hierarchies.txt"), "w")
  subset = np.random.randint(thought_vectors.shape[0], size=100)
  for i in range(100):
    current = subset[i]
    f.write(sentences[current] + " (%.4f)" % hyp_norms[current] + "\n")
    for j in range(10):
      current = _find_closest(current)
      if not current: break
      f.write(sentences[current] + " (%.4f)" % hyp_norms[current] + "\n")
    f.write("\n")
  f.close()


def elementwise_nearest_neighbours(thought_vectors, sentences, metric):
  """Find nearest neighbour to each given sentence.

  Args:
    thought_vectors: 2D numpy array containing the thought vectors.
    sentences: Corresponding string sentences.
    metric: To compute distances.
  """
  if metric == "hyperbolic":
    distances = _hyperbolic_distance(thought_vectors, thought_vectors)
  elif metric == "cosine":
    distances = - _cosine_similarity(thought_vectors, thought_vectors)
  else:
    distances = _sq_euclidean_distance(thought_vectors, thought_vectors)

  sorted_index = np.argsort(distances, axis=1)

  hyp_norms = np.squeeze(
      _hyperbolic_distance(thought_vectors,
                           np.zeros((1, thought_vectors.shape[1]))))

  f = tf.gfile.Open(os.path.join(FLAGS.output_dir,
                                 "sentence_nearest_neighbours.txt"), "w")
  for i in range(thought_vectors.shape[0]):
    f.write(sentences[i] + " (%.4f)" % hyp_norms[i] + "\n")
    for k in range(10):
      f.write(sentences[sorted_index[i,k+1]] +
              " (%.4f)" % hyp_norms[sorted_index[i,k+1]] + "\n")
    f.write("\n")
  f.close()


def save_test_sentence_norms(thought_vectors, sentences, metric, name):
  """Compute norms of the given thought vectors and save with the sentences.

  Args:
    thought_vectors: 2D numpy array containing the thought vectors in the same
      order as sentences.
    sentences: List of strings corresponding to the vectors above.
    metric: "hyperbolic" or "cosine" or None.
    name: File prefix for saving.
  """
  if metric == "hyperbolic":
    norms = np.squeeze(_hyperbolic_distance(
        thought_vectors, np.zeros((1, thought_vectors.shape[1]))))
  elif metric == "cosine":
    norms = np.linalg.norm(thought_vectors, axis=1)
  else:
    norms = np.linalg.norm(thought_vectors, axis=1)

  sorted_index = np.argsort(norms)

  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, name + "_sentence_norms.txt"),
                    "w")
  for ix in sorted_index:
    f.write(sentences[ix] + "\t" + "%.2f" % norms[ix] + "\n")
  f.close()


def save_target_probabilities(logits, targets, vocab):
  """Save the prediction probability of selected words.

  Args:
    logits: 2D Numpy array. Output logits per word.
    targets: 1D Numpy array. Target word to be predicted.
    vocab: Dict mapping words to their ids.
  """
  assert logits.shape[0] == targets.shape[0], (
      "Number of targets (%d) != Number of logits (%d)" % (targets.shape[0],
                                                           logits.shape[0]))
  types = []
  average_probabilities = []
  for typ in TARGET_TYPES:
    if typ not in vocab:
      tf.logging.info("%s OOV. Skipping.", typ)
      continue
    index = vocab[typ]

    occurrences = np.where(targets == index)[0]
    if occurrences.shape[0] == 0:
      tf.logging.info("No target matching %s. Skipping.", typ)
      continue

    prob_sum = 0.
    for ix in occurrences:
      scores = np.exp(logits[ix,:])
      probabilities = scores / np.sum(scores)
      prob_sum += probabilities[index]
    average_probabilities.append(prob_sum / occurrences.shape[0])
    types.append(typ)

  sorted_index = np.argsort(average_probabilities)

  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "target_probabilities.txt"),
                    "w")
  for ix in sorted_index:
    f.write(types[ix] + "\t" + "%.2f" % average_probabilities[ix] + "\n")
  f.close()


def plot_average_norm_by_length(embeddings, sentences, metric):
  """Plots the average embeddings norm for each length of sentences.

  Args:
    embeddings: 2D numpy array.
    sentences: List of string sentences corresponding to the embeddings.
    metric: "hyperbolic" or "cosine" or None.
  """
  if metric == "hyperbolic":
    norms = np.squeeze(_hyperbolic_distance(
        embeddings, np.zeros((1, embeddings.shape[1]))))
  elif metric == "cosine":
    norms = np.linalg.norm(embeddings, axis=1)
  else:
    norms = np.linalg.norm(embeddings, axis=1)
  lengths = [len(sentence.split()) for sentence in sentences]
  length_sum_norms = {}
  length_counts = {}
  for i in range(len(sentences)):
    if lengths[i] not in length_sum_norms:
      length_sum_norms[lengths[i]] = 0
      length_counts[lengths[i]] = 0
    length_sum_norms[lengths[i]] += norms[i]
    length_counts[lengths[i]] += 1

  tf.logging.info("Average Norm = %.4f",
                  float(sum(length_sum_norms.values())) /
                  sum(length_counts.values()))

  plot_x = []
  plot_y = []
  for length in length_sum_norms.keys():
    plot_x.append(length)
    plot_y.append(float(length_sum_norms[length]) / length_counts[length])

  fig, ax = plt.subplots()
  ax.plot(plot_x, plot_y)
  ax.set_xlabel('Sentence Length')
  ax.set_ylabel('Average Norm')
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "average_lengths.png"), "w")
  plt.savefig(f, dpi=1000)
  f.close()


def plot_embeddings(embeddings, labels, num_to_plot, name, metric, colors=None):
  """Plots a subset of the provided embeddings after projecting to 2D.

  If the embedding dimension > 2, these are first converted to 2 dimensions
  using T-SNE.

  Args:
    embeddings: 2D Numpy array containing embeddings one per row.
    labels: List of strings corresponding to annotations of the embeddings.
      If None no labels are annotated.
    num_to_plot: Number of embeddings to plot. These are randomly selected.
      Provide -1 to plot all embeddings
    name: Filename to save the plot in. .png extension will be appended.
    metric: Either "hyperbolic" or "cosine".

  Raises:
    ValueError: If the embeddings are less than 2 dimensions. A different
      visualization would be more appropriate in this case.
  """
  if embeddings.shape[1] < 2:
    raise ValueError("Embeddings must be at least 2 dimensional (found %d).",
                     embeddings.shape[1])
  elif embeddings.shape[1] == 2:
    tf.logging.info("Embeddings already 2 dimensional. No need for T-SNE.")
    embeddings_2d = embeddings
  else:
    tf.logging.info("Embeddings %d dimensional. Using T-SNE to project to 2 "
                    "dimensions.", embeddings.shape[1])
    model = TSNE(n_components=2, verbose=3, metric="precomputed")
    if metric == "hyperbolic":
      distances = _hyperbolic_distance(embeddings, embeddings)
    elif metric == "cosine":
      distances = - _cosine_similarity(embeddings, embeddings)
    else:
      distances = _sq_euclidean_distance(embeddings, embeddings)

    embeddings_2d = model.fit_transform(distances)
    tf.logging.info("Done.")

  if num_to_plot != -1:
    # Randomly sample subset to plot
    subset = np.random.randint(embeddings.shape[0], size=num_to_plot)
    embeddings_to_plot = embeddings_2d[subset, :]
    if colors:
      colors_to_plot = [colors[ii] for ii in subset]
    else:
      colors_to_plot = None
  else:
    subset = np.arange(embeddings.shape[0])
    embeddings_to_plot = embeddings_2d
    colors_to_plot = colors

  # Plot the subset
  fig, ax = plt.subplots()
  if colors_to_plot:
    ax.scatter(embeddings_to_plot[:,0], embeddings_to_plot[:,1],
               c=colors_to_plot)
  else:
    ax.scatter(embeddings_to_plot[:,0], embeddings_to_plot[:,1])
  if labels is not None:
    for i, ix in enumerate(subset):
      ax.annotate(labels[ix].decode(encoding="utf-8", errors="ignore"),
                  (embeddings_to_plot[i,0], embeddings_to_plot[i,1]),
                  fontsize=3)

  # Plot the unit circle for reference
  t = np.linspace(0, 2 * np.pi, 100)
  ax.plot(np.cos(t), np.sin(t), linewidth=1)

  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, name + ".png"),
                    "w")
  plt.savefig(f, dpi=1000)
  f.close()


def _create_decode_batch(sentences, vocab, length, pre):
  """Convert sentences into a batch for decoding.

  Args:
    sentences: List of strings to be decoded.
    vocab: A dict mapping tokens to indices.
    length: Minimum number of tokens to keep per instance.
    pre: Boolean indicating whether this batch corresponds to previous
      sentences or successor sentences.

  Returns:
    decode_ids: 2D padded numpy array containing token ids.
    decode_mask: 2D padded numpy array indicating where padding occurs.
  """
  token_lists = [sentence.lower().split() + ["<eos>"] for sentence in sentences]
  sentence_lengths = [len(sentence) for sentence in token_lists]

  decode_ids = np.zeros((len(token_lists), length), dtype="int32")
  decode_mask = np.zeros((len(token_lists), length), dtype="int32")
  for ix, sentence in enumerate(token_lists):
    to_fill = min(length, sentence_lengths[ix])
    for ii in range(to_fill):
      if pre:
        decode_ids[ix, -(ii+1)] = vocab.get(sentence[-(ii+1)], vocab["<unk>"])
        decode_mask[ix, -(ii+1)] = 1
      else:
        decode_ids[ix, ii] = vocab.get(sentence[ii], vocab["<unk>"])
        decode_mask[ix, ii] = 1

  return decode_ids, decode_mask


def _create_encode_batch(sentences, vocab):
  """Convert sentences into a batch.

  Args:
    vocab: A dict mapping tokens to indices.

  Returns:
    encode_ids: 2D padded numpy array containing token ids.
    encode_mask: 2D padded numpy array indicating where padding occurs.
  """
  token_lists = [sentence.lower().split() + ["<eos>"] for sentence in sentences]
  sentence_lengths = [len(sentence) for sentence in token_lists]

  encode_ids = np.zeros((len(token_lists), max(sentence_lengths)),
                        dtype="int32")
  encode_mask = np.zeros((len(token_lists), max(sentence_lengths)),
                         dtype="int32")
  for ix, sentence in enumerate(token_lists):
    encode_ids[ix, :sentence_lengths[ix]] = [vocab.get(token, vocab["<unk>"])
                                             for token in sentence]
    encode_mask[ix, :sentence_lengths[ix]] = 1

  return encode_ids, encode_mask


def _load_mr_dataset():
  pos_file = tf.gfile.Open(os.path.join(FLAGS.mr_data_dir, "rt-polarity.pos"))
  sentences = pos_file.read().splitlines()
  neg_file = tf.gfile.Open(os.path.join(FLAGS.mr_data_dir, "rt-polarity.neg"))
  sentences += neg_file.read().splitlines()
  return sentences


def main(unused_argv):
  # Create training directory if it doesn't already exist.
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.logging.info("Creating training directory: %s", FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Set up the model config.
  # Set up the model config.
  with tf.gfile.Open(
      os.path.join(FLAGS.checkpoint_path, "model_config.json")) as f:
    model_config_overrides = json.load(f)
  model_config = configuration.model_config(**model_config_overrides)
  tf.logging.info("model_config: %s",
                  json.dumps(model_config.values(), indent=2))

  # Read the vocabulary
  tf.logging.info("Reading vocabulary from: %s", FLAGS.vocab_file)
  vocab = tf.gfile.Open(FLAGS.vocab_file).read().splitlines()
  vocab_dict = {v: i for i,v in enumerate(vocab)}

  decode_length = model_config.context_window + model_config.condition_length

  tf.logging.info("Building training graph.")
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    tf.logging.info("Building model.")
    model = word_context_model.WordContextModel(
        model_config, mode="decode")
    model.build()
    saver = tf.train.Saver()
    # Create restore function
    restore_model, ckpt = _create_restore_fn(FLAGS.checkpoint_path, saver)

  # Fetch input embeddings from the model
  skip_thought_emb = tools.load_skip_thought_embeddings(
      ckpt, model_config.vocab_size, model_config.word_embedding_dim)

  with tf.Session(graph=g) as sess:
    restore_model(sess)
    #sess.run(tf.global_variables_initializer())
    #for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #  print(var.name)
    #skip_thought_emb = sess.run("word_embedding:0")

    # Fetch output embeddings from the model
    output_embeddings = sess.run(model.output_embeddings)
    #thought_projection_matrix = sess.run("decoder/thoughts_projection/weights:0")
    #context_projection_matrix = sess.run("decoder/context_projection/weights:0")

    # Encode test sentences
    encode_ids, encode_mask = _create_encode_batch(TEST_SENTENCES, vocab_dict)
    encode_emb = skip_thought_emb[encode_ids, :]
    test_thought_vectors, test_thought_vectors_pre = sess.run(
        [model.thought_vectors, model.thought_vectors_pre],
        feed_dict = {model.encode_emb: encode_emb,
                     model.encode_mask: encode_mask})

    # Fetch local context vectors from batch
    batch_sentences = tf.gfile.Open(FLAGS.sentences_file).read().splitlines()
    encode_ids, encode_mask = _create_encode_batch(
        batch_sentences[1:-1], vocab_dict)
    decode_pre_ids, decode_pre_mask = _create_decode_batch(
        batch_sentences[:-2], vocab_dict, decode_length, True)
    decode_post_ids, decode_post_mask = _create_decode_batch(
        batch_sentences[2:], vocab_dict, decode_length, False)
    encode_emb = skip_thought_emb[encode_ids, :]
    decode_pre_emb = skip_thought_emb[decode_pre_ids, :]
    decode_post_emb = skip_thought_emb[decode_post_ids, :]
    if model_config.decode_strategy == "conditional":
      (batch_outputs, batch_thought_vectors, batch_thought_vectors_pre,
       context_vectors, batch_logits, batch_targets,
       batch_target_mask) = sess.run(
           [model.encoder_outputs, model.thought_vectors,
            model.thought_vectors_pre, model.local_context_vectors,
            model.logits, model.targets, model.target_mask],
           feed_dict = {model.encode_emb: encode_emb,
                        model.encode_mask: encode_mask,
                        model.decode_pre_ids: decode_pre_ids,
                        model.decode_pre_emb: decode_pre_emb,
                        model.decode_pre_mask: decode_pre_mask,
                        model.decode_post_ids: decode_post_ids,
                        model.decode_post_emb: decode_post_emb,
                        model.decode_post_mask: decode_post_mask})
    else:
      tf.logging.info("Skipping local context vector computation since "
                      "decode_strategy != conditional.")
      context_vectors = None

  # Analyse outputs
  tf.logging.info("Analysing output norms")
  save_hyperbolic_norms(output_embeddings, vocab)
  save_euclidean_norms(output_embeddings, vocab)

  tf.logging.info("Finding nearest neighbours")
  save_nearest_neighbours(output_embeddings, vocab, SEED_TOKENS, 10,
                          model_config.logit_metric)

  tf.logging.info("Computing test sentence norms")
  save_test_sentence_norms(test_thought_vectors, TEST_SENTENCES,
                           model_config.logit_metric, "test")
  save_test_sentence_norms(batch_thought_vectors, batch_sentences,
                           model_config.logit_metric, "batch")

  tf.logging.info("Finding sentence nearest neighbours")
  elementwise_nearest_neighbours(test_thought_vectors, TEST_SENTENCES,
                                 model_config.logit_metric)

  tf.logging.info("Building sentence hierarchies")
  build_hierarchy(batch_thought_vectors, batch_sentences,
                  model_config.logit_metric)

  tf.logging.info("Computing target word probabilities")
  batch_targets_reshaped = np.reshape(batch_targets, (encode_ids.shape[0], -1))
  batch_target_mask_reshaped = np.reshape(batch_target_mask, (encode_ids.shape[0], -1))
  save_target_probabilities(batch_logits, batch_targets, vocab_dict)

  tf.logging.info("Saving word embeddings")
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "word_embeddings.npy"), "w")
  np.save(f, output_embeddings)
  f.close()
  tf.logging.info("Saving output embeddings")
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "batch_outputs.npy"), "w")
  np.save(f, batch_outputs)
  f.close()
  tf.logging.info("Saving thought vectors")
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "batch_thoughts.npy"), "w")
  np.save(f, batch_thought_vectors)
  f.close()
  tf.logging.info("Saving thought vectors before scaling")
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "batch_thoughts_pre.npy"), "w")
  np.save(f, batch_thought_vectors_pre)
  f.close()
  tf.logging.info("Saving thought vectors projection matrix")
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "thoughts_projection.npy"), "w")
  np.save(f, thought_projection_matrix)
  f.close()
  tf.logging.info("Saving context vectors projection matrix")
  f = tf.gfile.Open(os.path.join(FLAGS.output_dir, "context_projection.npy"), "w")
  np.save(f, context_projection_matrix)
  f.close()

  if FLAGS.plot:
    tf.logging.info("Plotting length distribution")
    if model_config.logit_metric == "cosine":
      rescaled_vectors = batch_thought_vectors / np.linalg.norm(
          batch_thought_vectors, axis=1, keepdims=True)
    else:
      rescaled_vectors = batch_thought_vectors
    plot_average_norm_by_length(rescaled_vectors, batch_sentences,
                                model_config.logit_metric)

    tf.logging.info("Plotting word embeddings")
    if model_config.logit_metric == "hyperbolic":
      reparameterized_emb = output_embeddings
    elif model_config.logit_metric == "cosine":
      reparameterized_emb = output_embeddings / np.linalg.norm(
          output_embeddings, axis=1, keepdims=True)
    else:
      reparameterized_emb = output_embeddings
    plot_embeddings(reparameterized_emb, vocab, 1000, "word_embeddings",
                    model_config.logit_metric)

    tf.logging.info("Plotting batch thought vectors")
    if model_config.logit_metric == "hyperbolic":
      rescaled_vectors = batch_thought_vectors
    elif model_config.logit_metric == "cosine":
      rescaled_vectors = batch_thought_vectors / np.linalg.norm(
          batch_thought_vectors, axis=1, keepdims=True)
    else:
      rescaled_vectors = batch_thought_vectors
    plot_embeddings(rescaled_vectors, batch_sentences, -1,
                    "batch_thought_vectors",
                    model_config.logit_metric)

    tf.logging.info("Plotting test thought vectors")
    if model_config.logit_metric == "hyperbolic":
      rescaled_vectors = test_thought_vectors
    elif model_config.logit_metric == "cosine":
      rescaled_vectors = test_thought_vectors / np.linalg.norm(
          test_thought_vectors, axis=1, keepdims=True)
    else:
      rescaled_vectors = test_thought_vectors
    plot_embeddings(rescaled_vectors, TEST_SENTENCES, -1,
                    "test_thought_vectors",
                    model_config.logit_metric)

    tf.logging.info("Plotting test thought vectors and word embeddings")
    # Randomly sample subset to plot
    subset = np.random.randint(output_embeddings.shape[0], size=500)
    rescaled_vectors = np.vstack([test_thought_vectors,
                                  output_embeddings[subset,:]])
    labels = TEST_SENTENCES + [vocab[ix] for ix in subset]
    colors = [0] * len(TEST_SENTENCES) + [1] * subset.shape[0]
    if model_config.logit_metric == "cosine":
      rescaled_vectors = rescaled_vectors / np.linalg.norm(
          rescaled_vectors, axis=1, keepdims=True)
    plot_embeddings(rescaled_vectors, labels, -1,
                    "all_vectors",
                    model_config.logit_metric,
                    colors=colors)

    if model_config.decode_strategy == "conditional":
      tf.logging.info("Plotting context vectors")
      if model_config.logit_metric == "hyperbolic":
        rescaled_vectors = context_vectors
      elif model_config.logit_metric == "cosine":
        rescaled_vectors = context_vectors / np.linalg.norm(
            context_vectors, axis=1, keepdims=True)
      else:
        rescaled_vectors = context_vectors
      plot_embeddings(rescaled_vectors, None, -1, "context_vectors",
                      model_config.logit_metric)


if __name__ == '__main__':
  tf.app.run()
