# NOTE: Run from root directory of repository

import argparse
import sys
import os
import time
import collections
from itertools import product

import numpy as np
import tensorflow as tf
import scipy.sparse

sys.path.insert(0, os.path.abspath(''))

import data
import utils
from lib_gcnn import graph, coarsening
from graph_cnn import GraphCNN
from train import train_and_test


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups", choices=data.AVAILABLE_DATASETS,
                    help="Dataset name (default: 20 Newsgroups)")
parser.add_argument("--vocab_size", type=int, default=None,
                    help="Vocabulary size (default: None [see data.py])")

args = parser.parse_args()


def run_experiment(x_train, y_train, x_valid, y_valid, embeddings, _num_edges, _filter_size, _num_features):
    # Feature graph parameters
    num_edges = _num_edges
    coarsening_levels = 0

    # Model parameters
    filter_name = "chebyshev"  # name of graph conv filter
    model_name = "gcnn_chebyshev"  # append filter name to model name
    filter_sizes = [_filter_size]  # filter sizes
    num_features = [_num_features]  # number of features per GCL
    pooling_sizes = [1]  # pooling sizes (1 (no pooling) or power of 2)
    fc_layers = []  # fully-connected layers

    # Training parameters
    learning_rate = 1e-3  # learning rate
    batch_size = 64  # batch size
    num_epochs = 100  # no. of training epochs

    # Regularization parameters
    dropout_keep_prob = 0.5  # dropout keep probability
    l2_reg_lambda = 0.0  # L2 regularization lambda

    # Feature Graph
    # ==================================================

    # Construct graph
    dist, idx = graph.distance_sklearn_metrics(embeddings, k=num_edges, metric="cosine")
    A = graph.adjacency(dist, idx)
    A = graph.replace_random_edges(A, 0)

    # Compute coarsened graphs
    graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
    laplacians = [graph.laplacian(A, normalized=True) for A in graphs]

    # Override filter sizes for non-param Fourier filter
    if filter_name == "fourier":
        filter_sizes = [l.shape[0] for l in laplacians]

    del dist, idx, A, graphs  # don't need these anymore

    # Reindex nodes to satisfy a binary tree structure
    x_train = scipy.sparse.csr_matrix(coarsening.perm_data(x_train.toarray(), perm))
    x_valid = scipy.sparse.csr_matrix(coarsening.perm_data(x_valid.toarray(), perm))

    # Training
    # ==================================================

    with tf.Graph().as_default():
        tf.set_random_seed(42)  # set random seed for consistent initialization(s)

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Init model
            gcnn = GraphCNN(filter_name=filter_name,
                            L=laplacians,
                            K=filter_sizes,
                            F=num_features,
                            P=pooling_sizes,
                            FC=fc_layers,
                            batch_size=batch_size,
                            num_vertices=x_train.shape[1],
                            num_classes=len(train.class_names),
                            l2_reg_lambda=l2_reg_lambda)

            # Convert sparse matrices to arrays
            x_train = np.squeeze([x_i.toarray() for x_i in x_train])
            x_valid = np.squeeze([x_i.toarray() for x_i in x_valid])

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.dataset, model_name,
                                                   timestamp))

            # Train and test model
            max_accuracy = train_and_test(sess, gcnn, x_train, y_train, x_valid, y_valid, learning_rate,
                                          batch_size, num_epochs, dropout_keep_prob, out_dir)

            return timestamp, max_accuracy


# Data Preparation
# ==================================================

train, test = data.load_dataset(args.dataset, out="tfidf", vocab_size=10000)
del test

x_train = train.data.astype(np.float32)
y_train = train.labels

# Split training set & validation set
validation_index = -1 * int(0.1 * float(len(y_train)))
x_train, x_valid = x_train[:validation_index], x_train[validation_index:]
y_train, y_valid = y_train[:validation_index], y_train[validation_index:]

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

num_edges_arr = [4, 8, 16]
filter_size_arr = [2, 4, 5]
num_features_arr = [8, 16, 32]
acc_dict = {}

for _num_edges, _filter_size, _num_features in product(num_edges_arr, filter_size_arr, num_features_arr):
    timestamp, max_accuracy = run_experiment(x_train, y_train, x_valid, y_valid, embeddings,
                                             _num_edges, _filter_size, _num_features)
    acc_dict["{} {} {}".format(_num_edges, _filter_size, _num_features)] = (max_accuracy, timestamp)
    with open("output.txt", "a") as file:
        file.write("{} {} {} {} {}\n".format(_num_edges, _filter_size, _num_features,
                                             max_accuracy, timestamp))

print(acc_dict)
