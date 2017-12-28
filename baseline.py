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
train.count_vectorize(stop_words="english")  # create term-document count matrix and vocabulary
orig_vocab_size = len(train.vocab)
train.remove_encoded_images()  # remove encoded images
train.keep_top_words(num_frequent_words)  # keep only the top words
train.remove_short_documents(nwords=5, vocab="selected")  # remove docs whose signal would be the zero vector
train.tfidf_normalize(norm="l1")  # transform count matrix into a normalized tf-idf matrix

print("Loading test data...")
test = data.Text20News(subset="test")
test.clean_text()
test.count_vectorize(vocabulary=train.vocab)
test.remove_encoded_images()
test.remove_short_documents(nwords=5, vocab="selected")
test.tfidf_normalize(norm="l1")

x_train = train.data_tfidf.astype(np.float32)
x_test = test.data_tfidf.astype(np.float32)
y_train = train.labels
y_test = test.labels

print("")
print("Vocabulary Size: {}".format(orig_vocab_size))
print("Vocabulary Size (Reduced): {}".format(len(train.vocab)))
print("Number of Classes: {}".format(len(train.class_names)))
print("Train/Test Split: {}/{}".format(len(y_train), len(y_test)))
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
