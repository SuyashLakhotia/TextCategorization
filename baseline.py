import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import data


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
print("Linear SVC Accuracy: {:.4f}".format(np.mean(predicted == y_test)))

# Multinomial Naive Bayes Classifier
bayes_clf = MultinomialNB(alpha=0.01)
bayes_clf.fit(x_train, y_train)
predicted = bayes_clf.predict(x_test)
print("Multinomial Naive Bayes Accuracy: {:.4f}".format(np.mean(predicted == y_test)))
