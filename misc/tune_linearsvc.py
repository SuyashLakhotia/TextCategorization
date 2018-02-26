import argparse
import sys
import os

# NOTE: Run from root directory of repository
sys.path.insert(0, os.path.abspath(''))

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import data
import utils


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups", choices=data.AVAILABLE_DATASETS,
                    help="Dataset name (default: 20 Newsgroups)")
parser.add_argument("--vocab_size", type=int, default=None,
                    help="Vocabulary size (default: None [see data.py])")
parser.add_argument("--out", type=str, default="tfidf", choices=["tfidf", "count"],
                    help="Type of document vectors (default: tfidf)")

parser.add_argument("--min", type=float, default=0)
parser.add_argument("--max", type=float, default=5)

args = parser.parse_args()


# Data Preparation
# ==================================================

train, test = data.load_dataset(args.dataset, out=args.out, vocab_size=args.vocab_size)

x_train = train.data.astype(np.float32)
x_test = test.data.astype(np.float32)
y_train = train.labels
y_test = test.labels

# Print information about the dataset
utils.print_data_info(train, x_train, x_test, y_train, y_test)

# To print at the end of script execution
data_str = "{{dataset: {}, format: '{}', vocab_size: {}}}".format(args.dataset, args.out, len(train.vocab))


# Training
# ==================================================

acc_dict = {}
C_arr = [float('%.1f' % i) for i in np.arange(args.min, args.max + 0.1, 0.1)]
for i in C_arr:
    if i <= 0:
        continue
    svm_clf = LinearSVC(C=i)
    svm_clf.fit(x_train, y_train)
    predicted = svm_clf.predict(x_test)
    svm_acc = np.mean(predicted == y_test)
    acc_dict[i] = svm_acc

print(data_str)
print(acc_dict)
