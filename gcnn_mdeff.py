import numpy as np
import tensorflow as tf
import scipy

from gcnn.graph import rescale_L


class GraphCNN(object):
    """
    A graph CNN for text classification. Composed of graph convolutional + max-pooling layer(s) and a 
    softmax layer.

    L = List of graph Laplacians.
    K = List of polynomial orders i.e. filter sizes (per filter)
    F = List of no. of features (per filter)
    p = List of pooling sizes (per filter)
    """

    def __init__(self, L, K, F, p, batch_size, num_vertices, num_classes, l2_reg_lambda):
        assert len(L) >= len(F) == len(K) == len(p)  # verify consistency w.r.t. the no. of GCLs
        assert np.all(np.array(p) >= 1)  # all pool sizes >= 1
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
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
        self.L = []
        for p_i in p:
            self.L.append(L[j])
            j += int(np.log2(p_i)) if p_i > 1 else 0
        L = self.L

        # Expand dims for convolution operation
        x = tf.expand_dims(self.input_x, 2)  # B x V x F=1

        # Graph convolutional + pooling layer(s)
        for i in range(len(K)):
            with tf.name_scope("conv-maxpool-{}".format(K[i])):
                F_in = int(x.get_shape()[2])
                W = tf.Variable(tf.truncated_normal([F_in * K[i], F[i]], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[1, 1, F[i]]), name="b")
                x = self.graph_conv_cheby(x, W, L[i], F[i], K[i]) + b
                x = tf.nn.relu(x)
                x = self.graph_max_pool(x, p[i])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(x, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            B, V, F = self.h_drop.get_shape()
            B, V, F = int(B), int(V), int(F)

            x = tf.reshape(self.h_drop, [B, V * F])
            W = tf.get_variable("W",
                                shape=[V * F, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(x, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.predictions = tf.cast(self.predictions, tf.int32)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def graph_conv_cheby(self, x, W, L, F_out, K):
        """
        Graph convolutional operation.
        """
        # B = Batch size
        # V = No. of vertices
        # F_in = No. of input features (per vertex)
        # F_out = No. of output features (per vertex)
        # K = Chebyshev polynomial order & support size
        B, V, F_in = x.get_shape()
        B, V, F_in = int(B), int(V), int(F_in)

        # Rescale Laplacian and store as a TF sparse tensor (copy to not modify the shared L)
        L = scipy.sparse.csr_matrix(L)
        L = rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        L = tf.cast(L, tf.float32)

        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])     # V x F_in x B
        x0 = tf.reshape(x0, [V, F_in * B])       # V x F_in*B
        x = tf.expand_dims(x0, 0)                # 1 x V x F_in*B

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)           # 1 x V x F_in*B
            return tf.concat([x, x_], axis=0)    # K x V x F_in*B
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # V x F_in*B
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, V, F_in, B])       # K x V x F_in x B
        x = tf.transpose(x, perm=[3, 1, 2, 0])   # B x V x F_in x K
        x = tf.reshape(x, [B * V, F_in * K])     # B*V x F_in*K

        # Compose linearly F_in features to get F_out features
        x = tf.matmul(x, W)                      # B*V x F_out
        x = tf.reshape(x, [B, V, F_out])         # B x V x F_out

        return x

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
