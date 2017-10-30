import numpy as np
import tensorflow as tf

from base.model import BasePWClassifier


class PWClassifier(BasePWClassifier):
    def __init__(self, n_users, n_items, lambda_p, lambda_q, n_factors=10, n_epochs=20, batch_size=1000,
                 learning_rate=0.001, random_state=None):
        super(PWClassifier, self).__init__(
            n_users, n_items, n_factors, n_epochs, batch_size, learning_rate, random_state
        )
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q

    def _build_graph(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        # uid, iid_i, iid_j
        X = tf.placeholder(tf.int32, shape=(None, 3), name="X")
        uids, iids_i, iids_j = tf.unstack(X, axis=1)

        y = tf.placeholder(tf.float32, shape=(None,), name="y")
        training = tf.placeholder_with_default(False, shape=(), name='training')

        p = tf.get_variable(
            "P",
            [self.n_users, self.n_factors],
            tf.float32,
            tf.variance_scaling_initializer,
            tf.contrib.layers.l2_regularizer(self.lambda_p)
        )
        q = tf.get_variable(
            "Q",
            [self.n_users, self.n_factors],
            tf.float32,
            tf.variance_scaling_initializer,
            tf.contrib.layers.l2_regularizer(self.lambda_q)
        )

        p_u = tf.nn.embedding_lookup(p, uids)
        q_i = tf.nn.embedding_lookup(q, iids_i)
        q_j = tf.nn.embedding_lookup(q, iids_j)

        with tf.name_scope("computation"):
            # x_ui - x_uj = p_u.q_i - p_u.q_j = p_u.(q_i - q_j)
            sub_ij = tf.subtract(q_i, q_j, name='sub_ij')
            dot = tf.reduce_sum(np.multiply(p_u, sub_ij), axis=1, name="dot")
            sigma = tf.sigmoid(dot, name="sigma")

        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=sigma)
        loss = tf.reduce_sum(x_entropy, name="loss")

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._graph_important_ops(X, y, training, training_op, loss, sigma, init, saver)

    def predict(self, X):
        return np.round(self.predict_proba(X))
