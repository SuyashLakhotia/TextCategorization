import subprocess

import numpy as np


def print_data_info(train, x_train, x_test, y_train, y_test):
    """
    Prints information about the train & test dataset.
    """
    print("")
    print("Original Vocabulary Size: {}".format(train.orig_vocab_size))
    print("Vocabulary Size (Reduced): {}".format(len(train.vocab)))
    print("")
    print("Train/Test Split: {}/{}".format(len(y_train), len(y_test)))
    print("Number of Classes: {}".format(len(train.class_names)))
    print("Train Class Split: {}".format(np.bincount(y_train)))
    print("Test Class Split: {}".format(np.bincount(y_test)))
    print("")
    print("x_train: {}".format(x_train.shape))
    print("x_test: {}".format(x_test.shape))
    print("y_train: {}".format(y_train.shape))
    print("y_test: {}".format(y_test.shape))
    print("")


def print_result(dataset, model_name, accuracy, data_str, timestamp, hyperparams="-", train_params=None):
    """
    Prints the record for results.csv.
    """
    latest_git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

    if train_params is None:
        train_str = "-"
    else:
        train_str = "{{learning_rate: {}, dropout: {}, l2_reg: {}, batch_size: {}, epochs: {}}}".format(
            train_params.learning_rate, train_params.dropout, train_params.l2, train_params.batch_size,
            train_params.epochs)

    print("")
    print("\"{}\",\"{}\",\"{:.9f}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\""
          .format(dataset, model_name, accuracy, data_str, hyperparams, train_str, "-", latest_git,
                  timestamp))
