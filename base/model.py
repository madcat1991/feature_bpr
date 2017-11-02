import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from metrics import accuracy_score_avg_by_users, bpr_auc_by_users


class BasePWClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_users, n_items, n_factors=10, n_epochs=20, batch_size=1000,
                 learning_rate=0.001, random_state=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.batch_size = batch_size
        self._session = None
        self._graph = None

    def close_session(self):
        if self._session:
            self._session.close()

    def _graph_important_ops(self, X, y, training, training_op, loss, y_proba, init, saver):
        self._X, self._y = X, y
        self._training = training
        self._training_op = training_op
        self._loss = loss
        self._y_proba = y_proba
        self._init, self._saver = init, saver

    def _build_graph(self, **kwargs):
        # input
        X = tf.placeholder(tf.int32, shape=(None, 3), name="X")
        y = tf.placeholder(tf.float32, shape=(None,), name="y")
        training = tf.placeholder_with_default(False, shape=(), name='training')

        # the graph should be here

        # init
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._graph_important_ops(X, y, training, None, None, None, init, saver)

    def fit(self, X, y, X_valid=None, y_valid=None, **kwargs):
        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(**kwargs)

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()

            for epoch in range(self.n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for i, rnd_indices in enumerate(np.array_split(rnd_idx, len(X) // self.batch_size)):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    sess.run(
                        self._training_op,
                        feed_dict={self._X: X_batch, self._y: y_batch, self._training: True}
                    )

                if X_valid is None:
                    X_valid = X_batch
                    y_valid = y_batch

                uids = X_valid[:, 0].reshape(-1)
                acc = accuracy_score_avg_by_users(y_valid, self.predict(X_valid), uids)
                auc = bpr_auc_by_users(y_valid, self.predict_proba(X_valid), uids)
                loss = sess.run(self._loss, feed_dict={self._X: X_valid, self._y: y_valid})
                print(
                    "%3s. loss=%.3f, acc=%.3f, auc=%.3f" %
                    (epoch, loss, acc, auc)
                )

    def save(self, path):
        self._saver.save(self._session, path)

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default():
            return self._y_proba.eval(feed_dict={self._X: X})
