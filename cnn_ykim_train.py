import argparse
import os
import time

import numpy as np
import tensorflow as tf

import data
import utils
from cnn_ykim import YKimCNN
from train import train_and_test


model_name = "cnn_ykim"


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups", choices=data.AVAILABLE_DATASETS,
                    help="Dataset name (default: 20 Newsgroups)")
parser.add_argument("--vocab_size", type=int, default=None,
                    help="Vocabulary size (default: None [see data.py])")

parser.add_argument("--seq_len", type=int, default=1000,
                    help="Sequence length for every pattern (default: 1000)")
parser.add_argument("--filter_widths", type=int, nargs="+", default=[3, 4, 5],
                    help="Filter widths (default: [3, 4, 5])")
parser.add_argument("--num_features", type=int, default=128,
                    help="No. of features per filter (default: 128)")
parser.add_argument("--fc_layers", type=int, nargs="*", default=None,
                    help="Fully-connected layers (default: None)")

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

# Pre-trained word embeddings
embedding_dim = 300  # dimensionality of embedding
embedding_file = "data/GoogleNews-vectors-negative300.bin"  # word embeddings file

# Model parameters
seq_len = args.seq_len  # sequence length for every pattern
filter_widths = args.filter_widths  # filter widths
num_features = args.num_features  # number of features per filter
fc_layers = args.fc_layers if args.fc_layers is not None else []  # fully-connected layers

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

train, test = data.load_dataset(args.dataset, out="word2ind", vocab_size=args.vocab_size, maxlen=seq_len)

x_train = train.data.astype(np.int32)
x_test = test.data.astype(np.int32)
y_train = train.labels
y_test = test.labels

# Correct sequence length if padding was overriden in data.py
seq_len = x_train.shape[1]

# Construct reverse lookup vocabulary
reverse_vocab = {w: i for i, w in enumerate(train.vocab)}

# Process Google News word2vec file (in a memory-friendly way) and store relevant embeddings
print("Loading pre-trained embeddings from {}...".format(embedding_file))
embeddings = data.load_word2vec(embedding_file, reverse_vocab, embedding_dim)

# Print information about the dataset
utils.print_data_info(train, x_train, x_test, y_train, y_test)

# To print for results.csv
data_str = "{{format: 'word2ind', vocab_size: {}, seq_len: {}}}".format(len(train.vocab), seq_len)


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                  log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Init model
        cnn = YKimCNN(sequence_length=seq_len,
                      num_classes=len(train.class_names),
                      vocab_size=len(train.vocab),
                      embedding_size=embedding_dim,
                      embeddings=embeddings,
                      filter_widths=filter_widths,
                      num_features=num_features,
                      fc_layers=fc_layers,
                      l2_reg_lambda=l2_reg_lambda)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.dataset, model_name, timestamp))

        # Train and test model
        max_accuracy = train_and_test(sess, cnn, x_train, y_train, x_test, y_test, learning_rate, batch_size,
                                      num_epochs, dropout_keep_prob, out_dir)

        # Output for results.csv
        hyperparams = "{{filter_widths: {}, num_features: {}, fc_layers: {}}}".format(filter_widths,
                                                                                      num_features,
                                                                                      fc_layers)
        utils.print_result(args.dataset, model_name, max_accuracy, data_str, timestamp, hyperparams, args,
                           args.notes)
