# Skip-Thought Vectors in TensorFlow

This directory contains a TensorFlow implementation of the
[Skip-Thought Vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf)
model.

The code in this directory is backed by the
[open sourced](https://github.com/tensorflow/models/tree/master/skip_thoughts)
implementation.

## Differences to the Open-Source Version

  * Supports distributed training (asynchronous or synchronous).
  * Uses the `tf.HParams` container for configurations.
  * Adds extra configuration options (e.g. optimizer).

## Differences to the Paper

  * This model does not condition at each timestep of the decoder. Encoded
    sentences are fed as the initial GRU state. The authors of the paper found
    that this was better after publication.
  * Layer normalization is applied as described in
    [this paper](https://arxiv.org/abs/1607.06450).
  * A learning-rate decay schedule is used by default, which speeds up training.

# Training a Model

### Prepare the Dataset

The model requires an SSTable of `tf.Example` protos containing the following
fields:

  * `encode`: The sentence to encode.
  * `decode_pre`: The sentence preceding `encode` in the original text.
  * `decode_post`: The sentence following `encode` in the original text.

Each sentence is an `Int64List` list of words ids. The word ids are integers in
the range `[0, vocab_size)` generated during preprocessing. The ids `0`
(end-of-sentence) and `1` (unknown word) are special ids.

Please see the open source TensorFlow SkipThoughts code for an example
preprocessing script.

### Launch Training

The following command will launch training and evaluation jobs for the
*Skip-Thoughts* model using the default configuration.

```shell
 $ bazel build -c opt \
    skip_thoughts_dist:train \
    skip_thoughts_dist:track_perplexity

 $ bazel-bin/skip_thoughts_dist/train \
    --input_file_pattern <input_file_pattern> \
    --train_dir <training_directory>
```

# Encoding Sentences

The
[`SkipThoughtsEncoder`](https://github.com/tensorflow/models/tree/master/skip_thoughts/skip_thoughts/skip_thoughts_encoder.py)
class can encode arbitrary sentences as skip-thought vectors using a trained
*Skip-Thoughts* model.

The
[`EncoderManager`](https://github.com/tensorflow/models/tree/master/skip_thoughts/skip_thoughts/encoder_manager.py)
class is a helpful wrapper for creating and initializing one or more
`SkipThoughtsEncoders`.

The following example uses data from the
[movie review dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
(specifically the
[sentence polarity dataset v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)).

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd

from skip_thoughts import configuration
from skip_thoughts import encoder_manager
```

```python
# Set paths to the model.
VOCAB_FILE = "/path/to/vocab.txt"
EMBEDDING_MATRIX_FILE = "/path/to/embeddings.npy"
CHECKPOINT_PATH = "/path/to/model.ckpt-9999"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
MR_DATA_DIR = "/dir/containing/mr/data"
```

```python
# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.
encoder = combined_encoder.CombinedEncoder()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)
```

```python
# Load the movie review dataset.
data = []
with gfile.Open(os.path.join(MR_DATA_DIR, 'rt-polarity.neg'), 'r') as f:
  data.extend([line.decode('latin-1').strip() for line in f])
with gfile.Open(os.path.join(MR_DATA_DIR, 'rt-polarity.pos'), 'r') as f:
  data.extend([line.decode('latin-1').strip() for line in f])
```

```python
# Generate Skip-Thought Vectors for each sentence in the dataset.
encodings = encoder.encode(data)
```

```python
# Define a helper function to generate nearest neighbors.
def get_nn(ind, num=10):
  encoding = encodings[ind]
  scores = sd.cdist([encoding], encodings, "cosine")[0]
  sorted_ids = np.argsort(scores)
  print("Sentence:")
  print("", data[ind])
  print("\nNearest neighbors:")
  for i in range(1, num + 1):
    print(" %d. %s (%.3f)" %
          (i, data[sorted_ids[i]], scores[sorted_ids[i]]))
```

```python
# Compute nearest neighbors of the first sentence in the dataset.
get_nn(0)
```

Output:

```
Sentence:
 simplistic , silly and tedious .

Nearest neighbors:
 1. trite , banal , cliched , mostly inoffensive . (0.247)
 2. banal and predictable . (0.253)
 3. witless , pointless , tasteless and idiotic . (0.272)
 5. grating and tedious . (0.299)
 6. idiotic and ugly . (0.330)
 7. black-and-white and unrealistic . (0.335)
 8. hopelessly inane , humorless and under-inspired . (0.335)
 9. shallow , noisy and pretentious . (0.340)
 10. . . . unlikable , uninteresting , unfunny , and completely , utterly inept . (0.346)
```
