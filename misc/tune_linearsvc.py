# NOTE: Run from root directory of repository

import argparse
import sys
import os
import collections
import time

sys.path.insert(0, os.path.abspath(''))

import numpy as np
from scipy.sparse import vstack
from sklearn.svm import LinearSVC

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

parser.add_argument("--test", action="store_false", dest="validation",
                    help="Include this flag if models should be tuned on the test set instead.")
parser.set_defaults(validation=True)

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

if args.validation:
    # Split training set & validation set
    validation_index = -1 * int(0.1 * float(len(y_train)))
    x_train, x_valid = x_train[:validation_index], x_train[validation_index:]
    y_train, y_valid = y_train[:validation_index], y_train[validation_index:]
else:
    x_valid, y_valid = [], []

# Print information about the dataset
print("")
print("Original Vocabulary Size: {}".format(train.orig_vocab_size))
print("Vocabulary Size (Reduced): {}".format(len(train.vocab)))
print("")
print("Train/Validation/Test Split: {}/{}/{}".format(len(y_train), len(y_valid), len(y_test)))
print("Number of Classes: {}".format(len(train.class_names)))
print("Train Class Split: {}".format(collections.Counter(y_train)))
print("Validation Class Split: {}".format(collections.Counter(y_valid)))
print("Test Class Split: {}".format(collections.Counter(y_test)))
print("")

# To print at the end of script execution
data_str = "{{dataset: {}, format: '{}', vocab_size: {}}}".format(args.dataset, args.out, len(train.vocab))


# Training
# ==================================================

# Generate C values to test [min, max, 0.1]
C_arr = [float('%.1f' % i) for i in np.arange(args.min, args.max + 0.1, 0.1)]

# Train & test models with different hyperparameter values
acc_dict = {}
for i in C_arr:
    if i <= 0:
        continue
    svm_clf = LinearSVC(C=i)
    svm_clf.fit(x_train, y_train)
    if args.validation:
        predicted = svm_clf.predict(x_valid)
        svm_acc = np.mean(predicted == y_valid)
    else:
        predicted = svm_clf.predict(x_test)
        svm_acc = np.mean(predicted == y_test)
    acc_dict[i] = svm_acc
    print("C {:.2f}: {:g}".format(i, svm_acc))

print(acc_dict)
print("")

# Concatenate training set & validation set to form original train set
if args.validation:
    x_train = vstack((x_train, x_valid))
    y_train = np.concatenate((y_train, y_valid), axis=0)

# Get optimized hyperparameter
max_C = max(acc_dict.keys(), key=(lambda key: acc_dict[key]))

# Re-train & test model with chosen hyperparameter
svm_clf = LinearSVC(C=max_C)
svm_clf.fit(x_train, y_train)
predicted = svm_clf.predict(x_test)
svm_acc = np.mean(predicted == y_test)

# Print result of final model
utils.print_result(args.dataset, "linear_svc", svm_acc, data_str, str(int(time.time())),
                   hyperparams="{{C: {}}}".format(max_C))
