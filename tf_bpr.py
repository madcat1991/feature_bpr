import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score


class TFBPR(BaseEstimator, RegressorMixin):
    def __init__(self, n_users, n_items, n_factors, lambda_, batch_size=1000, learning_rate=0.001, random_state=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.batch_size = batch_size
        self._session = None
        self._graph = None

    def close_session(self):
        if self._session:
            self._session.close()

    def _build_graph(self):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        # uid, iid_i, iid_j
        X = tf.placeholder(tf.int32, shape=(None, 3), name="X")
        # {-1, 1}
        y = tf.placeholder(tf.float32, shape=(None,), name="y")

        uids, iids_i, iids_j = tf.unstack(X, axis=1)

        p = tf.get_variable(
            "P",
            [self.n_users, self.n_factors],
            tf.float32,
            tf.variance_scaling_initializer,
            tf.contrib.layers.l2_regularizer(self.lambda_)
        )
        q = tf.get_variable(
            "Q",
            [self.n_users, self.n_factors],
            tf.float32,
            tf.variance_scaling_initializer,
            tf.contrib.layers.l2_regularizer(self.lambda_)
        )

        p_u = tf.nn.embedding_lookup(p, uids)
        q_i = tf.nn.embedding_lookup(q, iids_i)
        q_j = tf.nn.embedding_lookup(q, iids_j)

        with tf.name_scope("computation"):
            # x_ui - x_uj = p_u.q_i - p_u.q_j = p_u.(q_i - q_j)
            sub_ij = tf.subtract(q_i, q_j, name='sub_ij')
            mult = tf.multiply(p_u, sub_ij, name="mult")
            dot = tf.reduce_sum(mult, axis=1, name="dot")
            sigma = tf.sigmoid(dot, name="sigma")
            ln = tf.log(sigma, name="ln")
            ln_sum = tf.reduce_sum(ln)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_op = optimizer.minimize(-ln_sum)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._y_pred, self._ln_sum = sigma, ln_sum
        self._training_op = training_op
        self._init, self._saver = init, saver

    def fit(self, X, y, n_epochs):
        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()

            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for i, rnd_indices in enumerate(np.array_split(rnd_idx, len(X) // self.batch_size)):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    sess.run(self._training_op, feed_dict=feed_dict)

                ln_sum, y_pred = \
                    sess.run([self._ln_sum, self._y_pred], feed_dict={self._X: X_batch, self._y: y_batch})
                auc = roc_auc_score(y_batch, y_pred)
                print("%s.\tLast training batch: ln_sum=%.3f, auc=%.3f" % (epoch, ln_sum, auc))

    def save(self, path):
        self._saver.save(self._session, path)

    def predict(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._y_pred.eval(feed_dict={self._X: X})
