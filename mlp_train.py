import argparse
import os
import time

import numpy as np
import tensorflow as tf

import data
import utils
from mlp import MLP
from train import train_and_test


model_name = "mlp"


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups", choices=data.AVAILABLE_DATASETS,
                    help="Dataset name (default: 20 Newsgroups)")
parser.add_argument("--vocab_size", type=int, default=None,
                    help="Vocabulary size (default: None [see data.py])")
parser.add_argument("--out", type=str, default="tfidf", choices=["tfidf", "count"],
                    help="Type of document vectors (default: tfidf)")

parser.add_argument("--layers", type=int, nargs="*",
                    help="No. of units in fully-connected layers (default: None)")

parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization lambda (default: 0.0)")

parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
parser.add_argument("--epochs", type=int, default=200, help="No. of epochs (default: 200)")

parser.add_argument("--notes", type=str, default=None,
                    help="Any notes to add to the results.csv output (default: None)")

args = parser.parse_args()


# Parameters
# ==================================================

# Model parameters
layers = args.layers if args.layers is not None else []  # number of units in fully-connected layers

# Training parameters
learning_rate = args.learning_rate  # learning rate
batch_size = args.batch_size  # batch size
num_epochs = args.epochs  # no. of training epochs

# Regularization parameters
dropout_keep_prob = args.dropout  # dropout keep probability
l2_reg_lambda = args.l2  # L2 regularization lambda

# Misc. parameters
allow_soft_placement = True  # allow device soft device placement i.e. fall back on available device
log_device_placement = False  # log placement of operations on devices


# Data Preparation
# ==================================================

train, test = data.load_dataset(args.dataset, out=args.out, vocab_size=args.vocab_size)

x_train = train.data.astype(np.float32)
x_test = test.data.astype(np.float32)
y_train = train.labels
y_test = test.labels

# Print information about the dataset
utils.print_data_info(train, x_train, x_test, y_train, y_test)

# To print for results.csv
data_str = "{{format: '{}', vocab_size: {}}}".format(args.out, len(train.vocab))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                  log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        mlp = MLP(vocab_size=len(train.vocab),
                  num_classes=len(train.class_names),
                  layers=layers,
                  l2_reg_lambda=l2_reg_lambda)

        # Convert sparse matrices to arrays
        x_train = x_train.toarray()
        x_test = x_test.toarray()

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.dataset, model_name, timestamp))

        # Train and test model
        max_accuracy = train_and_test(sess, mlp, x_train, y_train, x_test, y_test, learning_rate, batch_size,
                                      num_epochs, dropout_keep_prob, out_dir)

        # Output for results.csv
        hyperparams = "{{layers: {}}}".format(layers)
        utils.print_result(args.dataset, model_name, max_accuracy, data_str, timestamp, hyperparams, args,
                           args.notes)
