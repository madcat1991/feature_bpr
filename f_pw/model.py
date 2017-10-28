import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score


class FPWClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_users, n_items, n_features, n_factors=10, lambda_=0.01, n_epochs=20, batch_size=1000,
                 learning_rate=0.001, random_state=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features

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

    def _build_graph(self, item_feature_m):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        # uid, iid_i, iid_j
        X = tf.placeholder(tf.int32, shape=(None, 3), name="X")
        uids, iids_i, iids_j = tf.unstack(X, axis=1)

        y = tf.placeholder(tf.float32, shape=(None,), name="y")

        # item-feature matrix
        if_m = tf.constant(item_feature_m, tf.float32, name="if_m")

        # users representation
        p = tf.get_variable(
            "P",
            [self.n_users, self.n_factors],
            tf.float32,
            tf.variance_scaling_initializer,
            tf.contrib.layers.l2_regularizer(self.lambda_)
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
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            activation=tf.nn.elu,
            name="W"
        )

        with tf.name_scope("computation"):
            dot = tf.reduce_sum(np.multiply(p_u, df_W), axis=1, name="dot")
            prediction = tf.sigmoid(dot, name="prediction")

        # loss = tf.nn.l2_loss(tf.subtract(prediction, y), name='loss')
        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction)
        loss = tf.reduce_sum(x_entropy, name="loss")

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y = X, y
        self._loss = loss
        self._y_pred = prediction
        self._training_op = training_op
        self._init, self._saver = init, saver

    def fit(self, X, y, item_feature_m):
        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(item_feature_m)

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()

            for epoch in range(self.n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for i, rnd_indices in enumerate(np.array_split(rnd_idx, len(X) // self.batch_size)):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})

                loss, y_pred = sess.run(
                    [self._loss, self._y_pred],
                    feed_dict={self._X: X_batch, self._y: y_batch}
                )
                auc = roc_auc_score(y_batch, y_pred)
                print("%3s. Last training batch: loss=%.3f, auc=%.3f" % (epoch, loss, auc))

    def save(self, path):
        self._saver.save(self._session, path)

    def predict(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default():
            return self._y_pred.eval(feed_dict={self._X: X})
