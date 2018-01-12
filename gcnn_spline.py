import numpy as np
import tensorflow as tf
import scipy

import lib_gcnn.graph as graph


class GCNN_Spline(object):
    """
    A graph CNN for text classification. Composed of graph convolutional + max-pooling layer(s) and a 
    softmax layer.

    L = List of graph Laplacians.
    K = List of filter sizes.
    F = List of no. of features (per filter).
    P = List of pooling sizes (per filter).
    FC = List of fully-connected layers.

    Code: Adapted from https://github.com/mdeff/cnn_graph
    """

    def __init__(self, L, K, F, P, FC, batch_size, num_vertices, num_classes, l2_reg_lambda):
        assert len(L) >= len(F) == len(K) == len(P)  # verify consistency w.r.t. the no. of GCLs
        assert np.all(np.array(P) >= 1)  # all pool sizes >= 1
        p_log2 = np.where(np.array(P) > 1, np.log2(P), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # all pool sizes > 1 should be powers of 2
        assert len(L) >= 1 + np.sum(p_log2)  # enough coarsening levels for pool sizes

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [batch_size, num_vertices], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [batch_size], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of L2 regularization loss
        l2_loss = tf.constant(0.0)

        # Keep the useful Laplacians only (may be zero)
        M_0 = L[0].shape[0]
        j = 0
        L_tmp = []
        for p_i in P:
            L_tmp.append(L[j])
            j += int(np.log2(p_i)) if p_i > 1 else 0
        L = L_tmp

        # Expand dims for convolution operation
        x = tf.expand_dims(self.input_x, 2)  # B x V x F=1

        # Graph convolutional + pooling layer(s)
        for i in range(len(K)):
            with tf.variable_scope("conv-maxpool-{}-{}".format(i, K[i])):
                B, V, F_in = x.get_shape()
                B, V, F_in = int(B), int(V), int(F_in)
                W = tf.Variable(tf.truncated_normal([K[i], F[i] * F_in], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[1, 1, F[i]]), name="b")
                x = self.spline(x, W, L[i], K[i], F[i]) + b
                x = tf.nn.relu(x)
                x = self.graph_max_pool(x, P[i])

        # Add dropout
        with tf.variable_scope("dropout"):
            x = tf.nn.dropout(x, self.dropout_keep_prob)

        # Reshape x for fully-connected layers
        with tf.variable_scope("reshape"):
            B, V, F = x.get_shape()
            B, V, F = int(B), int(V), int(F)
            x = tf.reshape(x, [B, V * F])

        # Add fully-connected layers (if any)
        for i, num_units in enumerate(FC):
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

    def spline(self, x, W, L, K, F_out):
        """
        Graph convolution operation.
        """
        B, V, F_in = x.get_shape()
        B, V, F_in = int(B), int(V), int(F_in)

        # Fourier basis
        lamb, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)  # V x V

        # Spline basis
        basis = self.bspline_basis(K, lamb, degree=3)  # V x K
        basis = tf.constant(basis, dtype=tf.float32)

        # Weight multiplication
        W = tf.matmul(basis, W)  # V x F_out*F_in
        W = tf.reshape(W, [V, F_out, F_in])

        return self.filter_in_fourier(x, L, K, F_out, U, W)

    def filter_in_fourier(self, x, L, K, F_out, U, W):
        # TODO: B x F x V would avoid the permutations
        B, V, F_in = x.get_shape()
        B, V, F_in = int(B), int(V), int(F_in)
        x = tf.transpose(x, perm=[1, 2, 0])  # V x F_in x B

        # Transform to Fourier domain
        x = tf.reshape(x, [V, F_in * B])  # V x F_in*B
        x = tf.matmul(U, x)  # V x F_in*B
        x = tf.reshape(x, [V, F_in, B])  # V x F_in x B

        # Filter
        x = tf.matmul(W, x)  # for each feature
        x = tf.transpose(x)  # B x F_out x V
        x = tf.reshape(x, [B * F_out, V])  # B*F_out x V

        # Transform back to graph domain
        x = tf.matmul(x, U)  # B*F_out x V
        x = tf.reshape(x, [B, F_out, V])  # B x F_out x V

        return tf.transpose(x, perm=[0, 2, 1])  # B x V x F_out

    def graph_max_pool(self, x, p):
        """
        Graph max pooling operation. p must be 1 or a power of 2.
        """
        if p > 1:
            x = tf.expand_dims(x, 3)   # B x V x F x 1
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[1, p, 1, 1], padding="SAME")
            return tf.squeeze(x, [3])  # B x V/p x F
        else:
            return x

    def bspline_basis(self, K, x, degree=3):
        """
        Return the B-spline basis.

        K: Number of control points.
        x: Evaluation points or number of evenly distributed evaluation points.
        degree: Degree of the spline. Cubic spline by default.
        """
        if np.isscalar(x):
            x = np.linspace(0, 1, x)

        # Evenly distributed knot vectors
        kv1 = x.min() * np.ones(degree)
        kv2 = np.linspace(x.min(), x.max(), K - degree + 1)
        kv3 = x.max() * np.ones(degree)
        kv = np.concatenate((kv1, kv2, kv3))

        # Cox-DeBoor recursive function to compute one spline over x
        def cox_deboor(k, d):
            # Test for end conditions, the rectangular degree zero spline
            if (d == 0):
                return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

            denom1 = kv[k + d] - kv[k]
            term1 = 0
            if denom1 > 0:
                term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

            denom2 = kv[k + d + 1] - kv[k + 1]
            term2 = 0
            if denom2 > 0:
                term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

            return term1 + term2

        # Compute basis for each point
        basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
        basis[-1, -1] = 1
        return basis
