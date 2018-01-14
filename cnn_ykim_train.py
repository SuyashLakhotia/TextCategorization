import argparse
import os
import time

import numpy as np
import tensorflow as tf

import data
from cnn_ykim import CNN_YKim
from train import train_and_test


model_name = "cnn_ykim"


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups", help="Dataset name")

parser.add_argument("--seq_len", type=int, default=10000, help="Sequence length for every pattern")
parser.add_argument("--filter_heights", type=int, nargs="+", default=[3, 4, 5], help="Filter heights")
parser.add_argument("--num_features", type=int, default=128, help="No. of features per filter")

parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=200, help="No. of epochs")

parser.add_argument("--dropout", type=float, default=0.5, help="Dropout keep probability")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization lambda")

args = parser.parse_args()


# Parameters
# ==================================================

# Pre-trained word embeddings
embedding_dim = 300  # dimensionality of embedding
embedding_file = "data/GoogleNews-vectors-negative300.bin"  # word embeddings file

# Model parameters
seq_len = args.seq_len  # sequence length for every pattern
filter_heights = args.filter_heights  # filter heights
num_features = args.num_features  # number of features per filter

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
train, test = data.load_dataset(dataset, out="word2ind", maxlen=seq_len)

x_train = train.data.astype(np.int32)
x_test = test.data.astype(np.int32)
y_train = train.labels
y_test = test.labels

# Correct sequence length if padding was overriden in data.py
seq_len = x_train.shape[1]

# Construct reverse lookup vocabulary.
reverse_vocab = {w: i for i, w in enumerate(train.vocab)}

# Process Google News word2vec file (in a memory-friendly way) and store relevant embeddings.
print("Loading pre-trained embeddings from {}...".format(embedding_file))
embeddings = data.load_word2vec(embedding_file, reverse_vocab, embedding_dim)

print("")
print("Vocabulary Size: {}".format(train.orig_vocab_size))
print("Vocabulary Size (Reduced): {}".format(len(train.vocab)))
print("Max. Document Length: {}".format(seq_len))
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
        # Init model
        cnn = CNN_YKim(sequence_length=seq_len,
                       num_classes=len(train.class_names),
                       vocab_size=len(train.vocab),
                       embedding_size=embedding_dim,
                       embeddings=embeddings,
                       filter_heights=filter_heights,
                       num_features=num_features,
                       l2_reg_lambda=l2_reg_lambda)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dataset, model_name, timestamp))

        # Train and test model
        max_accuracy = train_and_test(sess, cnn, x_train, y_train, x_test, y_test, learning_rate, batch_size,
                                      num_epochs, dropout_keep_prob, out_dir)

        # Output for results.csv
        hyperparams = "{{seq_len: {}, filter_heights: {}, num_features: {}}}".format(
            seq_len, filter_heights, num_features)
        data.print_result(dataset, model_name, max_accuracy, hyperparams, timestamp)
