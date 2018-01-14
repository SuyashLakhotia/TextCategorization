import argparse

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import data


# Parse Arguments
# ==================================================

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="20 Newsgroups",
                    help="Dataset name (default: 20 Newsgroups)")

args = parser.parse_args()


# Data Preparation
# ==================================================

dataset = args.dataset
train, test = data.load_dataset(dataset, out="tfidf", norm="l1")

x_train = train.data.astype(np.float32)
x_test = test.data.astype(np.float32)
y_train = train.labels
y_test = test.labels

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
data.print_result(dataset, "Linear SVC", svm_acc)
data.print_result(dataset, "Multinomial Naive Bayes", bayes_acc)
