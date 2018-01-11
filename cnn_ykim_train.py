import os
import time
import subprocess

import numpy as np
import tensorflow as tf

import data
from cnn_ykim import TextCNN
from train import train_and_test


model_name = "cnn_ykim"


# Parameters
# ==================================================

# Pre-trained word embeddings
embedding_dim = 300  # dimensionality of embedding
embedding_file = "data/GoogleNews-vectors-negative300.bin"  # word embeddings file

# Model parameters
seq_len = 10000  # sequence length for every pattern
filter_heights = [3, 4, 5]  # filter heights
num_features = 128  # number of features per filter

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
train.preprocess_train(out="word2ind", maxlen=seq_len)

print("Loading test data...")
test = data.Text20News(subset="test")
test.preprocess_test(train_vocab=train.vocab, out="word2ind", maxlen=seq_len)

x_train = train.data_word2ind.astype(np.int32)
x_test = test.data_word2ind.astype(np.int32)
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
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model_name, timestamp))

        # Train and test model
        max_accuracy = train_and_test(sess, cnn, x_train, y_train, x_test, y_test, learning_rate, batch_size,
                                      num_epochs, dropout_keep_prob, out_dir)

        # Output for results.csv
        hyperparams = "{{seq_len: {}, filter_heights: {}, num_features: {}}}".format(
            seq_len, filter_heights, num_features)
        latest_git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
        print("\"{}\",\"{}\",\"{:.9f}\",\"{}\",\"{}\"".format(model_name, hyperparams, max_accuracy,
                                                              latest_git, timestamp))
