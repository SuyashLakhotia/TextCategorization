import argparse
import time
import os
import pickle

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

parser.add_argument("--model", type=str, default="all", choices=["all", "linear_svc", "multinomial_nb"],
                    help="Model(s) to run on dataset.")

parser.add_argument("--LSVC-C", type=float, default=1.0, dest="C",
                    help="LinearSVC: Penalty parameter C of the error term.")
parser.add_argument("--MNB-a", type=float, default=0.01, dest="alpha",
                    help="MultinomialNB: Additive smoothing parameter (0 for no smoothing).")

parser.add_argument("--no-save", action="store_false", dest="save",
                    help="Include this flag if models should not be pickled.")
parser.set_defaults(save=True)

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

# To print for results.csv
data_str = "{{format: '{}', vocab_size: {}}}".format(args.out, len(train.vocab))


# Training
# ==================================================

timestamp = str(int(time.time()))

# Linear Support Vector Classifier
if args.model == "all" or args.model == "linear_svc":
    svm_clf = LinearSVC(C=args.C)
    svm_clf.fit(x_train, y_train)
    predicted = svm_clf.predict(x_test)
    svm_acc = np.mean(predicted == y_test)
    utils.print_result(args.dataset, "linear_svc", svm_acc, data_str, timestamp)

# Multinomial Naive Bayes Classifier
if args.model == "all" or args.model == "multinomial_nb":
    bayes_clf = MultinomialNB(alpha=args.alpha)
    bayes_clf.fit(x_train, y_train)
    predicted = bayes_clf.predict(x_test)
    bayes_acc = np.mean(predicted == y_test)
    utils.print_result(args.dataset, "multinomial_nb", bayes_acc, data_str, timestamp)

# Save models as pickles
if args.save:
    if args.model == "all" or args.model == "linear_svc":
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.dataset, "linear_svc",
                                               timestamp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pickle.dump(svm_clf, open(out_dir + "/pickle.pkl", "wb"))

    if args.model == "all" or args.model == "multinomial_nb":
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", args.dataset, "multinomial_nb",
                                               timestamp))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pickle.dump(bayes_clf, open(out_dir + "/pickle.pkl", "wb"))
