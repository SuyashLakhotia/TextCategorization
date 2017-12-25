import numpy as np
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import data


# Parameters
# ==================================================

# Preprocessing parameters
num_frequent_words = 10000  # number of frequent words to retain


# Data Preparation
# ==================================================

print("Loading training data...")
train = data.Text20News(subset="train")  # load data
train.remove_short_documents(nwords=20, vocab="full")  # remove documents < 20 words in length
train.clean_text()  # tokenize & clean text
train.vectorize(stop_words="english")  # create term-document matrix and vocabulary
train.remove_encoded_images()  # remove encoded images
train.keep_top_words(num_frequent_words)  # keep only the top words
train.remove_short_documents(nwords=5, vocab="selected")  # remove documents whose signal would be the zero vector
train.normalize(norm="l1")  # normalize data

print("Loading test data...")
test = data.Text20News(subset="test")
test.clean_text()  # tokenize & clean text
test.vectorize(vocabulary=train.vocab)  # create term-document matrix using train.vocab
test.remove_encoded_images()  # remove encoded images
test.remove_short_documents(nwords=5, vocab="selected")  # remove documents whose signal would be the zero vector
test.normalize(norm="l1")  # normalize data

x_train = train.data.astype(np.float32)
x_test = test.data.astype(np.float32)
y_train = train.labels
y_test = test.labels

print("Vocabulary Size: {}".format(len(train.vocab)))
print("Number of Classes: {}".format(len(train.class_names)))
print("Train/Test Split: {}/{}".format(len(y_train), len(y_test)))


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
