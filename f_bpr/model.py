import numpy as np
import tensorflow as tf

from base.model import BasePWClassifier


class FBPR(BasePWClassifier):
    def __init__(self, n_users, n_items, n_features, lambda_p, lambda_w, n_factors=10,
                 n_epochs=20, batch_size=1000, learning_rate=0.001, random_state=None):
        super(FBPR, self).__init__(
            n_users, n_items, n_factors, n_epochs, batch_size, learning_rate, random_state
        )
        self.lambda_p = lambda_p
        self.lambda_w = lambda_w
        self.n_features = n_features

    def _build_graph(self, item_feature_m):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        # uid, iid_i, iid_j
        X = tf.placeholder(tf.int32, shape=(None, 3), name="X")
        uids, iids_i, iids_j = tf.unstack(X, axis=1)

        y = tf.placeholder(tf.float32, shape=(None,), name="y")
        training = tf.placeholder_with_default(False, shape=(), name='training')

        # item-feature matrix
        if_m = tf.constant(item_feature_m, tf.float32, name="if_m")

        # users representation
        p = tf.get_variable(
            "P",
            [self.n_users, self.n_factors],
            tf.float32,
            tf.variance_scaling_initializer,
            tf.contrib.layers.l2_regularizer(self.lambda_p)
        )

        # x_ui - x_uj = p_u.q_i - p_u.q_j = p_u.(f_i.W - f_j.W) = p_u.d_f.W
        p_u = tf.nn.embedding_lookup(p, uids)
        with tf.name_scope("features"):
            iids_i_features = tf.gather(if_m, iids_i)
            iids_j_features = tf.gather(if_m, iids_j)
            delta_f = tf.subtract(iids_i_features, iids_j_features, "delta_f")

        df_W = tf.layers.dense(
            delta_f,
            self.n_factors,
            use_bias=False,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_w),
            name="W"
        )

        logits = tf.reduce_sum(tf.multiply(p_u, df_W), axis=1, name="logits")
        y_proba = tf.nn.sigmoid(logits, name="y_proba")

        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_sum(x_entropy, name="loss")

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._graph_important_ops(X, y, training, training_op, loss, y_proba, init, saver)

    def predict(self, X):
        return np.round(self.predict_proba(X))
