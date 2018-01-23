import argparse
import os
import time

import numpy as np
import tensorflow as tf
import scipy.sparse

import data
import utils
from lib_gcnn import graph, coarsening
from graph_cnn import GraphCNN
from train import train_and_test


model_name = "gcnn_"


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups",
                    help="Dataset name (default: 20 Newsgroups)")
parser.add_argument("--vocab_size", type=int, default=None,
                    help="Vocabulary size (default: None [see data.py])")

parser.add_argument("--num_edges", type=int, default=16, help="No. of edges in feature graph (default: 16)")
parser.add_argument("--coarsening_levels", type=int, default=0,
                    help="Coarsening levels for feature graph (default: 0)")

parser.add_argument("--filter_name", type=str, default="chebyshev",
                    help="Name of graph convolutional filter (default: chebyshev)")
parser.add_argument("--filter_sizes", type=int, nargs="+", default=[5], help="Filter sizes (default: [5])")
parser.add_argument("--num_features", type=int, nargs="+", default=[32],
                    help="No. of features per GCL (default: [32])")
parser.add_argument("--pooling_sizes", type=int, nargs="+", default=[1], help="Pooling sizes (default: [1])")
parser.add_argument("--fc_layers", type=int, nargs="*", help="Fully-connected layers (default: None)")

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

# Feature graph parameters
num_edges = args.num_edges
coarsening_levels = args.coarsening_levels

# Model parameters
filter_name = args.filter_name  # name of graph conv filter
model_name += filter_name  # append filter name to model name
filter_sizes = args.filter_sizes  # filter sizes
num_features = args.num_features  # number of features per GCL
pooling_sizes = args.pooling_sizes  # pooling sizes (1 (no pooling) or power of 2)
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

train, test = data.load_dataset(args.dataset, out="tfidf", vocab_size=args.vocab_size, norm="l1")

x_train = train.data.astype(np.float32)
x_test = test.data.astype(np.float32)
y_train = train.labels
y_test = test.labels

# Construct reverse lookup vocabulary
reverse_vocab = {w: i for i, w in enumerate(train.vocab)}

# Process Google News word2vec file (in a memory-friendly way) and store relevant embeddings
print("Loading pre-trained embeddings from {}...".format(embedding_file))
embeddings = data.load_word2vec(embedding_file, reverse_vocab, embedding_dim)

# Print information about the dataset
utils.print_data_info(train, x_train, x_test, y_train, y_test)

# To print for results.csv
data_str = "{{format: 'tfidf', vocab_size: {}}}".format(len(train.vocab))


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

del embeddings, dist, idx, A, graphs  # don't need these anymore

# Reindex nodes to satisfy a binary tree structure
x_train = scipy.sparse.csr_matrix(coarsening.perm_data(x_train.toarray(), perm))
x_test = scipy.sparse.csr_matrix(coarsening.perm_data(x_test.toarray(), perm))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                  log_device_placement=log_device_placement)
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
        # TODO: Is there a workaround for this? Doesn't seem memory efficient.
        # TODO: https://github.com/tensorflow/tensorflow/issues/342#issuecomment-160354041
        # TODO: https://github.com/tensorflow/tensorflow/issues/342#issuecomment-273463729
        # TODO: https://stackoverflow.com/questions/37001686/using-sparsetensor-as-a-trainable-variable
        x_train = np.squeeze([x_i.toarray() for x_i in x_train])
        x_test = np.squeeze([x_i.toarray() for x_i in x_test])

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.dataset, model_name, timestamp))

        # Train and test model
        max_accuracy = train_and_test(sess, gcnn, x_train, y_train, x_test, y_test, learning_rate,
                                      batch_size, num_epochs, dropout_keep_prob, out_dir)

        # Output for results.csv
        hyperparams = "{{num_edges: {}, coarsening_levels: {}, filter_sizes: {}, num_features: {}, pooling_sizes: {}, fc_layers: {}}}".format(
            num_edges, coarsening_levels, filter_sizes, num_features, pooling_sizes, fc_layers)
        utils.print_result(args.dataset, model_name, max_accuracy, data_str, timestamp, hyperparams, args,
                           args.notes)
