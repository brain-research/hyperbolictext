# HyperText: Models for learning Text Representations

This directory contains TensorFlow source code for learning
embeddings of text sequences in an unsupervised manner. This is a preliminary
implementation with more changes (and documentation) forthcoming.

## Contact
***Code authors:*** Bhuwan Dhingra, Chris Shallue

***Pull requests and issues:*** @bdhingra, @cshallue

## Getting Started

This code is backed by the open-source TensorFlow implementation of the
Skip-Thought vectors model availabel
[here](https://github.com/tensorflow/models/tree/master/skip_thoughts). Hence,
the first step is to download that repository and copy the relevant
`skip_thoughts` directory to this repository.

```shell
TENSORFLOW_MODELS_DIR="${HOME}/tensorflow_models/"

git clone https://github.com/tensorflow/models.git "$TENSORFLOW_MODELS_DIR"

cp -r "${TENSORFLOW_MODELS_DIR}/skip_thoughts/skip_thoughts/" .
```

Follow the instructions in `${TENSORFLOW_MODELS_DIR}/skip_thoughts/README.md`
and ensure that all the source files from that repository are working properly.

## Distributed Skip Thoughts

The [skip_thoughts_dist](skip_thoughts_dist) directory contains training and
validation scripts which can be used to run the TensorFlow Skip Thoughts model
in a distributed setting. These can be used in the following manner:

### Training

```shell
#!/bin/bash

bazel build -c opt //skip_thoughts_dist:train

bazel-bin/skip_thoughts_dist/train \
  --input_file_pattern <tfrecord_files> \
  --nosync_replicas \
  --train_dir <path_to_save_models>
```

### Validation

Run the validation script for tracking perplexity in a separate process. You may
want to pass the `CUDA_VISIBLE_DEVICES=''` flag to avoid using the GPU for this
script.

```shell
#!/bin/bash

bazel build -c opt //skip_thoughts_dist:track_perplexity

CUDA_VISIBLE_DEVICES='' bazel-bin/skip_thoughts_dist/track_perplexity \
  --input_file_pattern <tfrecord_files> \
  --checkpoint_dir <path_with_saved_models> \
  --eval_dir <directory_to_log_eval_summaries>
```

## Word Context Models

Word Context models decode each word in a specified window around the source
sentence separately as opposed to the full sequence decoder in the original
SkipThoughts model. There are several different options on the particular design
of the encoder; see the [configuration](word_context/configuration.py) file for
details.

In particular, setting `decode_strategy = "conditional"` and `logit_metric =
"hyperbolic"` will train hyperbolic sentence embeddings.

Training and validation scripts can be run in exactly the same manner as for
distributed SkipThoughts above.

### Training

```shell
#!/bin/bash

bazel build -c opt //word_context:train

bazel-bin/word_context/train \
  --input_file_pattern <tfrecord_files> \
  --nosync_replicas \
  --train_dir <path_to_save_models>
```

### Validation

```shell
#!/bin/bash

bazel build -c opt //word_context:track_perplexity

CUDA_VISIBLE_DEVICES='' bazel-bin/word_context/track_perplexity \
  --input_file_pattern <tfrecord_files> \
  --checkpoint_dir <path_with_saved_models> \
  --eval_dir <directory_to_log_eval_summaries>
```

### Evaluation

Once a word context model is trained, it can be evaluated as a feature extractor
on downstream tasks. In addition to the tasks described in the SkipThoughts
paper, we also provide code to evaluate on NLI tasks. See the
[readme](nli/README.md) file in the `nli` directory for details on how this
works.

To run an evaluation on a trained Word Context model (including expanding the
vocabulary as described in the SkipThoughts paper):

```shell
#!/bin/bash

bazel build -c opt //word_context:evaluate_trec

CUDA_VISIBLE_DEVICES='' bazel-bin/word_context/evaluate_trec \
  --checkpoint_dir <path_to_trained_model> \
  --skip_thought_vocab_file <path_to_training_vocabulary_file> \
  --word2vec_embedding_file <path_to_word2vec embeddings> \
  --word2vec_vocab_file <path_to_word2vec_vocabulary> \
  --eval_tasks CR,MR,SUBJ,MPQA,MultiNLI,SNLI \
  --data_dir <path_to_skipthoughts_eval_data> \
  --multinli_dir <path_to_multinli_data> \
  --nouse_norm \
  --nouse_eos \
  --nli_eval_method mlp \
  --nli_eval_dir <path_to_store_NLI_eval_checkpoints>
```

## Note

This is not an official Google product.
