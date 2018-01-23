import tensorflow as tf


class MLP(object):
    """
    A multi-layer perceptron made up of a number of fully-connected layers followed by a softmax layer.
    """

    def __init__(self, vocab_size, num_classes, layers, l2_reg_lambda):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, vocab_size], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.train_flag = tf.placeholder(tf.bool, name="train_flag")
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name="dropout_keep_prob")

        # Keeping track of L2 regularization loss
        l2_loss = tf.constant(0.0)

        # Create fully-connected layers
        x = self.input_x
        for i, num_units in enumerate(layers):
            with tf.variable_scope("fc-{}-{}".format(i, num_units)):
                W = tf.get_variable("W",
                                    shape=[x.get_shape().as_list()[1], num_units],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_units]), name="b")

                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)

                x = tf.nn.xw_plus_b(x, W, b)
                x = tf.nn.relu(x)
                x = tf.nn.dropout(x, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable("W",
                                shape=[x.get_shape().as_list()[1], num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(x, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predictions = tf.cast(self.predictions, tf.int32)

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
