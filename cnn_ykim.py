import tensorflow as tf


class YKimCNN(object):
    """
    A CNN architecture for text classification. Composed of an embedding layer, followed by parallel 
    convolutional + max-pooling layer(s) and a softmax layer.

    Paper: https://arxiv.org/abs/1408.5882
    Code: Adapted from https://github.com/dennybritz/cnn-text-classification-tf
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, embeddings, filter_widths,
                 num_features, fc_layers, l2_reg_lambda):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.train_flag = tf.placeholder(tf.bool, name="train_flag")
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name="dropout_keep_prob")

        # Keeping track of L2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.variable_scope("embedding"):
            if embeddings is None:
                embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                            name="embeddings")
            else:
                embedding_mat = tf.Variable(embeddings, name="embeddings")
            self.embedded_x = tf.nn.embedding_lookup(embedding_mat, self.input_x)
            self.embedded_x = tf.cast(self.embedded_x, tf.float32)

        # Convolution + max-pool layer for each filter size
        pooled_outputs = []
        for i in range(len(filter_widths)):
            with tf.variable_scope("conv-maxpool-{}".format(i)):
                conv_x = self.embedded_x
                with tf.variable_scope("conv-{}-{}".format(filter_widths[i], num_features)):
                    # Convolution layer
                    filter_shape = [filter_widths[i], embedding_size, num_features]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    conv_x = tf.nn.conv1d(value=conv_x,
                                          filters=W,
                                          stride=1,
                                          padding="VALID",
                                          name="conv")

                    # Add bias & apply non-linearity
                    b = tf.Variable(tf.constant(0.1, shape=[num_features]), name="b")
                    conv_x = tf.nn.relu(tf.nn.bias_add(conv_x, b), name="relu")
                with tf.variable_scope("maxpool"):
                    # Max-pooling over the outputs
                    pooled = tf.reduce_max(conv_x, axis=1, name="pool")
                    pooled_outputs.append(pooled)

        # Combine all the pooled features
        with tf.variable_scope("concat"):
            num_features_total = num_features * len(filter_widths)
            self.x = tf.concat(pooled_outputs, -1)

        # Add dropout
        with tf.variable_scope("dropout"):
            self.x = tf.nn.dropout(self.x, self.dropout_keep_prob)

        # Fully-connected layers, if any
        for i, num_units in enumerate(fc_layers):
            with tf.variable_scope("fc-{}-{}".format(i, num_units)):
                W = tf.get_variable("W",
                                    shape=[self.x.get_shape().as_list()[1], num_units],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_units]), name="b")

                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)

                self.x = tf.nn.xw_plus_b(self.x, W, b)
                self.x = tf.nn.relu(self.x)
                self.x = tf.nn.dropout(self.x, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_features_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.x, W, b, name="scores")
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
