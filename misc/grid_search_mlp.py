# NOTE: Run from root directory of repository

import argparse
import sys
import os
import time
import collections

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(''))

import data
from mlp import MLP
from train import train_and_test


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups", choices=data.AVAILABLE_DATASETS,
                    help="Dataset name (default: 20 Newsgroups)")
parser.add_argument("--vocab_size", type=int, default=None,
                    help="Vocabulary size (default: None [see data.py])")

parser.add_argument("--epochs", type=int, default=100, help="No. of epochs (default: 100)")

parser.add_argument("--test", action="store_false", dest="validation",
                    help="Include this flag if models should be tuned on the test set instead.")
parser.set_defaults(validation=True)

args = parser.parse_args()


def run_experiment(x_train, y_train, x_valid, y_valid, embeddings, _layers):
    # Model parameters
    model_name = "mlp"
    layers = _layers

    # Training parameters
    learning_rate = 1e-3  # learning rate
    batch_size = 64  # batch size
    num_epochs = args.epochs  # no. of training epochs

    # Regularization parameters
    dropout_keep_prob = 0.5  # dropout keep probability
    l2_reg_lambda = 0.0  # L2 regularization lambda

    # Training
    # ==================================================

    with tf.Graph().as_default():
        tf.set_random_seed(42)  # set random seed for consistent initialization(s)

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Init model
            mlp = MLP(vocab_size=len(train.vocab),
                      num_classes=len(train.class_names),
                      layers=layers,
                      l2_reg_lambda=l2_reg_lambda)

            # Convert sparse matrices to arrays
            x_train = x_train.toarray()
            x_valid = x_valid.toarray()

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.dataset, model_name,
                                                   timestamp))

            # Train and test model
            max_accuracy = train_and_test(sess, mlp, x_train, y_train, x_valid, y_valid, learning_rate,
                                          batch_size, num_epochs, dropout_keep_prob, out_dir)

            return timestamp, max_accuracy


# Data Preparation
# ==================================================

train, test = data.load_dataset(args.dataset, out="tfidf", vocab_size=10000)

x_train = train.data.astype(np.float32)
y_train = train.labels

if args.validation:
    del test  # don't need this anymore

    # Split training set & validation set
    validation_index = -1 * int(0.1 * float(len(y_train)))
    x_train, x_valid = x_train[:validation_index], x_train[validation_index:]
    y_train, y_valid = y_train[:validation_index], y_train[validation_index:]
else:
    x_valid = test.data.astype(np.float32)
    y_valid = test.labels

# Construct reverse lookup vocabulary
reverse_vocab = {w: i for i, w in enumerate(train.vocab)}

# Pre-trained word embeddings
embedding_dim = 300  # dimensionality of embedding
embedding_file = "data/GoogleNews-vectors-negative300.bin"  # word embeddings file

# Process Google News word2vec file (in a memory-friendly way) and store relevant embeddings
print("Loading pre-trained embeddings from {}...".format(embedding_file))
embeddings = data.load_word2vec(embedding_file, reverse_vocab, embedding_dim)

# Print information about the dataset
print("")
print("Original Vocabulary Size: {}".format(train.orig_vocab_size))
print("Vocabulary Size (Reduced): {}".format(len(train.vocab)))
print("")
print("Train/Validation Split: {}/{}".format(len(y_train), len(y_valid)))
print("Number of Classes: {}".format(len(train.class_names)))
print("Train Class Split: {}".format(collections.Counter(y_train)))
print("Validation Class Split: {}".format(collections.Counter(y_valid)))
print("")


# Grid Search
# ==================================================

layers_arr = [[], [100], [250], [500], [1000], [2000], [2000, 500], [2000, 1000]]
acc_dict = {}

for _layers in layers_arr:
    timestamp, max_accuracy = run_experiment(x_train, y_train, x_valid, y_valid, embeddings, _layers)
    acc_dict["{}".format(_layers)] = (max_accuracy, timestamp)
    with open("output_mlp.txt", "a") as file:
        file.write("{} {} {}\n".format(_layers, max_accuracy, timestamp))

print(acc_dict)
