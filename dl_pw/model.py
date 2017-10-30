import numpy as np
import tensorflow as tf

from base.model import BasePWClassifier


class DLPWClassifier(BasePWClassifier):
    def __init__(self, n_users, n_items, n_features, lambda_ol, lambda_hl, h_layers,
                 n_factors=10, n_epochs=20, batch_size=1000, dropout_rate=None,
                 batch_norm_momentum=None, activation=tf.nn.elu,
                 learning_rate=0.001, random_state=None):
        super(DLPWClassifier, self).__init__(
            n_users, n_items, n_factors, n_epochs, batch_size, learning_rate, random_state
        )
        self.n_features = n_features
        self.h_layers = h_layers
        self.lambda_ol = lambda_ol
        self.lambda_hl = lambda_hl
        self.dropout_rate = dropout_rate
        self.batch_norm_momentum = batch_norm_momentum
        self.activation = activation
        self.n_output = 2

    def _dnn(self, input_, training):
        last_output = input_
        for i, n_neurons in enumerate(self.h_layers):
            if self.dropout_rate:
                last_output = tf.layers.dropout(
                    last_output,
                    self.dropout_rate,
                    training=training
                )

            last_output = tf.layers.dense(
                last_output,
                n_neurons,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_hl),
                name="hidden-%d" % i
            )

            if self.batch_norm_momentum:
                last_output = tf.layers.batch_normalization(
                    last_output,
                    momentum=self.batch_norm_momentum,
                    training=training
                )

            last_output = self.activation(
                last_output,
                name="hidden_%s_out" % i
            )
        return last_output

    def _prerpare_input(self, p_u, iids_i_features, iids_j_features):
        return tf.concat([p_u, iids_i_features, iids_j_features], 1)

    def _build_graph(self, item_feature_m):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        # uid, iid_i, iid_j
        X = tf.placeholder(tf.int32, shape=(None, 3), name="X")
        uids, iids_i, iids_j = tf.unstack(X, axis=1)

        y = tf.placeholder(tf.int32, shape=(None,), name="y")
        training = tf.placeholder_with_default(False, shape=(), name='training')

        # item-feature matrix
        if_m = tf.constant(item_feature_m, tf.float32, name="if_m")

        # users representation
        p = tf.get_variable(
            "P",
            [self.n_users, self.n_factors],
            tf.float32,
            tf.variance_scaling_initializer,
            tf.contrib.layers.l2_regularizer(self.lambda_ol)
        )

        with tf.name_scope("features"):
            p_u = tf.nn.embedding_lookup(p, uids)
            iids_i_features = tf.gather(if_m, iids_i)
            iids_j_features = tf.gather(if_m, iids_j)

        input_f = self._prerpare_input(p_u, iids_i_features, iids_j_features)
        dnn_outputs = self._dnn(input_f, training)
        logits = tf.layers.dense(
            dnn_outputs,
            self.n_output,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_ol),
            name="outputs"
        )

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")

        y_proba = tf.nn.softmax(logits, name="y_proba")

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._graph_important_ops(X, y, training, training_op, loss, y_proba, init, saver)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class DLSubPWClassifier(DLPWClassifier):
    def _prerpare_input(self, p_u, iids_i_features, iids_j_features):
        delta_f = tf.subtract(iids_i_features, iids_j_features, "delta_f")
        return tf.concat([p_u, delta_f], 1)
