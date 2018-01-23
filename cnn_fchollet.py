import tensorflow as tf


class FCholletCNN(object):
    """
    A CNN architecture for text classification. Composed of an embedding layer followed by convolutional + 
    max-pooling layer(s), fully-connected layer(s) and a softmax layer.

    Blog Post: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    Code: Adapted from https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, embeddings, filter_widths,
                 num_features, pooling_sizes, fc_layers, l2_reg_lambda):
        assert len(filter_widths) == len(num_features) == len(pooling_sizes)

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
            embedded_x = tf.nn.embedding_lookup(embedding_mat, self.input_x)
            embedded_x = tf.cast(embedded_x, tf.float32)
            self.x = embedded_x

        # Create a convolution + max-pool layer for each filter size
        for i, filter_width in enumerate(filter_widths):
            with tf.variable_scope("conv-maxpool-{}-{}".format(i, filter_width)):
                if i == 0:
                    in_channels = embedding_size
                else:
                    in_channels = num_features[i - 1]
                filter_shape = [filter_width, in_channels, num_features[i]]

                # Convolution layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                conv = tf.nn.conv1d(value=self.x,
                                    filters=W,
                                    stride=1,
                                    padding="VALID",
                                    name="conv")

                # Add bias & apply non-linearity
                b = tf.Variable(tf.constant(0.1, shape=[num_features[i]]), name="b")
                self.x = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Max-pooling over the outputs
                self.x = tf.nn.pool(input=self.x,
                                    window_shape=[pooling_sizes[i]],
                                    strides=[pooling_sizes[i]],
                                    pooling_type="MAX",
                                    padding="VALID",
                                    name="pool")

        # Global max pooling
        with tf.variable_scope("global-maxpool"):
            self.x = tf.reduce_max(self.x, axis=1)

        # Create fully-connected layers, if any
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
                                shape=[self.x.get_shape().as_list()[1], num_classes],
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
