import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from metrics import accuracy_score_avg_by_users


class PWClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_users, n_items, n_factors=10, lambda_=0.01, n_epochs=20, batch_size=1000,
                 learning_rate=0.001, random_state=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
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
        uids, iids_i, iids_j = tf.unstack(X, axis=1)

        y = tf.placeholder(tf.float32, shape=(None,), name="y")

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
            dot = tf.reduce_sum(np.multiply(p_u, sub_ij), axis=1, name="dot")
            sigma = tf.sigmoid(dot, name="sigma")
            prediction = tf.round(sigma)

        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=sigma)
        loss = tf.reduce_sum(x_entropy, name="loss")

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._loss = loss
        self._y_prob = sigma
        self._y_predicted = prediction
        self._training_op = training_op
        self._init, self._saver = init, saver

    def fit(self, X, y):
        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()

            for epoch in range(self.n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for i, rnd_indices in enumerate(np.array_split(rnd_idx, len(X) // self.batch_size)):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})

                loss, y_pred = sess.run(
                    [self._loss, self._y_predicted],
                    feed_dict={self._X: X_batch, self._y: y_batch}
                )
                acc = accuracy_score_avg_by_users(y_batch, y_pred, X_batch[:, 0].reshape(-1))
                print("%3s. Last training batch: loss=%.3f, accuracy=%.3f" % (epoch, loss, acc))

    def save(self, path):
        self._saver.save(self._session, path)

    def predict(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default():
            return self._y_predicted.eval(feed_dict={self._X: X})

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default():
            return self._y_prob.eval(feed_dict={self._X: X})
