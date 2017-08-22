# Natural Language Inference Evaluation

These scripts are used to evaluate sentence encoders on Natural Language
Inference tasks, in particular the
[MultiNLI](http://www.nyu.edu/projects/bowman/nli/) and
[SNLI](https://nlp.stanford.edu/projects/snli/) datasets.

## Data Preparation

First prepare your data into `train`, `test` and `dev` splits. For each split
create a `.jsonl` (loose json) file containing one json object per line. The
json object should contain the following fields:

* `gold_label`: Ground truth label of relation between sentences (entailment /
  contradiction / nuetral).
* `sentence1`: Premise sentence string.
* `sentence2`: Hypothesis sentence string.
* `genre`: (Optional) genre to which the two sentences belong. If provided, the
  genre-wise results will be computed.

## Sentence Encoder

The sentence encoder is a python class instance implementing the `encode` method
for converting a list of text sentences into a list of numpy vectors. For an
example see `baseline_encoders.RandomEncoder`.


## Evaluation

To run an evaluation:

```python
from nli import eval_nli

# <nli_path_prefix> specifies the NLI data files. Expects
# <nli_path_prefix>_train.jsonl, <nli_path_prefix>_dev.jsonl,
# <nli_path_prefix>_test.jsonl

# eval_dir is the location to store intermediate model files. Only useful when
# method == 'mlp'

sentence_encoder = YourSentenceEncoder()
results = eval_nli.evaluate(
    sentence_encoder,
    nli_path_prefix,
    eval_dir,
    method='logistic')
```

The `method` argument to `evaluate` specifies the classifier to use for
evaluating. Allowed options are:

* `logistic`: To train a Logistic Regression classifier using the sklearn
  library. See
  [this](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  for details.
* `mlp`: This trains a Multi-Layer Perceptron with three hidden layers and an
  output layer (with softmax) for classification. The implementation uses
  TensorFlow and works best with a GPU. See the [code](mlp_classifier.py) for
  details.
