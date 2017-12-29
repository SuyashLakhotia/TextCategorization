import os
import datetime

import numpy as np
import tensorflow as tf
import sklearn.metrics

import data


def train_and_test(sess, model, x_train, y_train, x_test, y_test, learning_rate, batch_size, num_epochs,
                   dropout_keep_prob, out_dir, evaluate_every=100, checkpoint_every=100, num_checkpoints=5):
    print("Writing to {}\n".format(out_dir))

    # Define training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Test Summary Writer
    test_summary_dir = os.path.join(out_dir, "summaries", "test")
    test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

    # Checkpoint directory & saver
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        """
        A single training step.
        """
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.dropout_keep_prob: dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, model.loss,
                                                       model.accuracy],
                                                      feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: Step {}, Loss {:g}, Accuracy {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def test_step(x_test, y_test, writer=None):
        """
        Evaluates model on a test set.
        """
        # TODO: Hacky workaround to test model due to OOM errors / fixed batch size.
        step = 0
        size = x_test.shape[0]
        losses = 0
        predictions = np.empty(size)
        for begin in range(0, size, batch_size):
            end = begin + batch_size
            end = min([end, size])

            x_batch = np.zeros((batch_size, x_test.shape[1]))
            x_batch[:end - begin] = x_test[begin:end]

            y_batch = np.zeros(batch_size)
            y_batch[:end - begin] = y_test[begin:end]

            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: 1.0
            }
            step, batch_pred, batch_loss = sess.run([global_step, model.predictions, model.loss],
                                                    feed_dict)

            predictions[begin:end] = batch_pred[:end - begin]
            losses += batch_loss

        accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
        loss = losses * batch_size / size

        time_str = datetime.datetime.now().isoformat()
        print("{}: Step {}, Loss {:g}, Accuracy {:g}".format(time_str, step, loss, accuracy))

        summary = tf.Summary()
        summary.value.add(tag="loss_1", simple_value=loss)
        summary.value.add(tag="accuracy_1", simple_value=accuracy)

        if writer:
            writer.add_summary(summary, step)
        return accuracy

    # Generate batches
    batches = data.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

    # Maximum test accuracy
    max_accuracy = 0.0

    # Training loop
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            accuracy = test_step(x_test, y_test, writer=test_summary_writer)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
            print("Max. Test Accuracy: {:g}".format(max_accuracy))
            print("")
        if current_step % checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

    return max_accuracy
