import tensorflow as tf


class CNN_YKim(object):
    """
    A CNN architecture for text classification. Composed of an embedding layer, followed by parallel 
    convolutional + max-pooling layer(s) and a softmax layer.

    Paper: https://arxiv.org/abs/1408.5882
    Code: Adapted from https://github.com/dennybritz/cnn-text-classification-tf
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, embeddings, filter_heights,
                 num_features, l2_reg_lambda):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of L2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.variable_scope("embedding"):
            if embeddings is None:
                self.embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                                 name="embeddings")
            else:
                self.embedding_mat = tf.Variable(embeddings, name="embeddings")
            self.embedded_x = tf.nn.embedding_lookup(self.embedding_mat, self.input_x)
            self.embedded_x = tf.expand_dims(self.embedded_x, -1)  # expand for .conv2d
            self.embedded_x = tf.cast(self.embedded_x, tf.float32)

        # Create a convolution + max-pool layer for each filter size (filter_height x embedding_size)
        pooled_outputs = []
        for i, filter_height in enumerate(filter_heights):
            with tf.variable_scope("conv-maxpool-{}-{}".format(i, filter_height)):
                # Convolution layer
                filter_shape = [filter_height, embedding_size, 1, num_features]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_features]), name="b")
                conv = tf.nn.conv2d(self.embedded_x,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")

                # Apply non-linearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, sequence_length - filter_height + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        with tf.variable_scope("reshape"):
            num_features_total = num_features * len(filter_heights)
            h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(h_pool, [-1, num_features_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_features_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
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
