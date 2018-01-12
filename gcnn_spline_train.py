import os
import time
import subprocess

import numpy as np
import tensorflow as tf
import scipy.sparse

import data
from lib_gcnn import graph, coarsening
from gcnn_spline import GCNN_Spline
from train import train_and_test


model_name = "gcnn_spline"


# Parameters
# ==================================================

# Pre-trained word embeddings
embedding_dim = 300  # dimensionality of embedding
embedding_file = "data/GoogleNews-vectors-negative300.bin"  # word embeddings file

# Feature graph parameters
num_edges = 16
coarsening_levels = 0

# Model parameters
filter_sizes = [5]  # filter sizes
num_features = [32]  # number of features per GCL
pooling_sizes = [1]  # pooling sizes (1 (no pooling) or power of 2)
fc_layers = []  # fully-connected layers

# Training parameters
learning_rate = 1e-3
batch_size = 64
num_epochs = 200

# Regularization parameters
dropout_keep_prob = 0.5  # dropout keep probability
l2_reg_lambda = 0.0  # L2 regularization lambda

# Misc. parameters
allow_soft_placement = True  # allow device soft device placement i.e. fall back on available device
log_device_placement = False  # log placement of operations on devices


# Data Preparation
# ==================================================

print("Loading training data...")
train = data.Text20News(subset="train")
train.preprocess_train(out="tfidf", norm="l1")

print("Loading test data...")
test = data.Text20News(subset="test")
test.preprocess_test(train_vocab=train.vocab, out="tfidf", norm="l1")

x_train = train.data_tfidf.astype(np.float32)
x_test = test.data_tfidf.astype(np.float32)
y_train = train.labels
y_test = test.labels

# Construct reverse lookup vocabulary.
reverse_vocab = {w: i for i, w in enumerate(train.vocab)}

# Process Google News word2vec file (in a memory-friendly way) and store relevant embeddings.
print("Loading pre-trained embeddings from {}...".format(embedding_file))
embeddings = data.load_word2vec(embedding_file, reverse_vocab, embedding_dim)

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


# Feature Graph
# ==================================================

# Construct graph
dist, idx = graph.distance_sklearn_metrics(embeddings, k=num_edges, metric="cosine")
A = graph.adjacency(dist, idx)
A = graph.replace_random_edges(A, 0)

# Compute coarsened graphs
graphs, perm = coarsening.coarsen(A, levels=coarsening_levels, self_connections=False)
laplacians = [graph.laplacian(A, normalized=True) for A in graphs]

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
        gcnn = GCNN_Spline(L=laplacians, K=filter_sizes, F=num_features, P=pooling_sizes, FC=fc_layers,
                           batch_size=batch_size,
                           num_vertices=len(train.vocab),
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_name, timestamp))

        # Train and test model
        max_accuracy = train_and_test(sess, gcnn, x_train, y_train, x_test, y_test, learning_rate,
                                      batch_size, num_epochs, dropout_keep_prob, out_dir)

        # Output for results.csv
        hyperparams = "{{num_edges: {}, coarsening_levels: {}, filter_sizes: {}, num_features: {}, pooling_sizes: {}, fc_layers: {}, dropout: {}}}".format(
            num_edges, coarsening_levels, filter_sizes, num_features, pooling_sizes, fc_layers, dropout_keep_prob)
        latest_git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        print("\"{}\",\"{}\",\"{:.9f}\",\"{}\",\"{}\"".format(model_name, hyperparams, max_accuracy,
                                                              latest_git, timestamp))
