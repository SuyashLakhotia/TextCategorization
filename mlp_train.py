import argparse
import os
import time

import numpy as np
import tensorflow as tf

import data
from mlp import MLP
from train import train_and_test


model_name = "mlp"


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups",
                    help="Dataset name (default: 20 Newsgroups)")

parser.add_argument("--layers", type=int, nargs="*",
                    help="No. of units in fully-connected layers (default: None)")

parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization lambda (default: 0.0)")

parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
parser.add_argument("--epochs", type=int, default=200, help="No. of epochs (default: 200)")

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

dataset = args.dataset
train, test = data.load_dataset(dataset, out="tfidf", norm="l1")

x_train = train.data.astype(np.float32)
x_test = test.data.astype(np.float32)
y_train = train.labels
y_test = test.labels

print("")
print("Vocabulary Size: {}".format(train.orig_vocab_size))
print("Vocabulary Size (Reduced): {}".format(len(train.vocab)))
print("Number of Classes: {}".format(len(train.class_names)))
print("Train/Test Split: {}/{}".format(len(y_train), len(y_test)))
print("")
print("x_train: {}".format(x_train.shape))
print("x_test: {}".format(x_test.shape))
print("y_train: {}".format(y_train.shape))
print("y_test: {}".format(y_test.shape))
print("")


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

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dataset, model_name, timestamp))

        # Convert sparse matrices to arrays
        x_train = x_train.toarray()
        x_test = x_test.toarray()

        # Train and test model
        max_accuracy = train_and_test(sess, mlp, x_train, y_train, x_test, y_test, learning_rate, batch_size,
                                      num_epochs, dropout_keep_prob, out_dir)

        # Output for results.csv
        hyperparams = "{{layers: {}}}".format(layers)
        data.print_result(dataset, model_name, max_accuracy, timestamp, hyperparams, args)
