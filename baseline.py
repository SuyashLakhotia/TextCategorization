import argparse
import time

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import data
import utils


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups",
                    help="Dataset name (default: 20 Newsgroups)")
parser.add_argument("--vocab_size", type=int, default=None,
                    help="Vocabulary size (default: None [see data.py])")

args = parser.parse_args()


# Data Preparation
# ==================================================

train, test = data.load_dataset(args.dataset, out="tfidf", vocab_size=args.vocab_size, norm="l1")

x_train = train.data.astype(np.float32)
x_test = test.data.astype(np.float32)
y_train = train.labels
y_test = test.labels

# Print information about the dataset
utils.print_data_info(train, x_train, x_test, y_train, y_test)

# To print for results.csv
data_str = "{{format: 'word2ind', vocab_size: {}}}".format(len(train.vocab))


# Training
# ==================================================

# Linear Support Vector Classifier
svm_clf = LinearSVC()
svm_clf.fit(x_train, y_train)
predicted = svm_clf.predict(x_test)
svm_acc = np.mean(predicted == y_test)

# Multinomial Naive Bayes Classifier
bayes_clf = MultinomialNB(alpha=0.01)
bayes_clf.fit(x_train, y_train)
predicted = bayes_clf.predict(x_test)
bayes_acc = np.mean(predicted == y_test)

# Output for results.csv
timestamp = str(int(time.time()))
utils.print_result(args.dataset, "Linear SVC", svm_acc, data_str, timestamp)
utils.print_result(args.dataset, "Multinomial Naive Bayes", bayes_acc, data_str, timestamp)
