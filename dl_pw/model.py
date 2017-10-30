import numpy as np
import tensorflow as tf

from f_pw.model import FPWClassifier


class DLPWClassifier(FPWClassifier):
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
            tf.contrib.layers.l2_regularizer(self.lambda_)
        )

        with tf.name_scope("features"):
            p_u = tf.nn.embedding_lookup(p, uids)
            iids_i_features = tf.gather(if_m, iids_i)
            iids_j_features = tf.gather(if_m, iids_j)

        input_f = tf.concat([p_u, iids_i_features, iids_j_features], 1)

        dropout_rate = 0.5
        n_outputs = 2

        d0 = tf.layers.dense(
            input_f,
            200,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            activation=tf.nn.elu,
            name="d0"
        )
        d0_drop = tf.layers.dropout(d0, dropout_rate, training=training)

        d1 = tf.layers.dense(
            d0_drop,
            200,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            activation=tf.nn.elu,
            name="d1"
        )
        d1_drop = tf.layers.dropout(d1, dropout_rate, training=training)

        d2 = tf.layers.dense(
            d1_drop,
            200,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            activation=tf.nn.elu,
            name="d2"
        )
        d2_drop = tf.layers.dropout(d2, dropout_rate, training=training)

        logits = tf.layers.dense(
            d2_drop,
            n_outputs,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            # activation=tf.nn.elu,
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
        self._X, self._y = X, y
        self._training = training
        self._loss = loss
        self._y_prob = y_proba
        self._training_op = training_op
        self._init, self._saver = init, saver

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class DLSubPWClassifier(FPWClassifier):
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
            tf.contrib.layers.l2_regularizer(self.lambda_)
        )

        with tf.name_scope("features"):
            p_u = tf.nn.embedding_lookup(p, uids)
            iids_i_features = tf.gather(if_m, iids_i)
            iids_j_features = tf.gather(if_m, iids_j)
            delta_f = tf.subtract(iids_i_features, iids_j_features, "delta_f")

        input_f = tf.concat([p_u, delta_f], 1)

        dropout_rate = 0.5
        n_outputs = 2

        d0 = tf.layers.dense(
            input_f,
            200,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            activation=tf.nn.elu,
            name="d0"
        )
        d0_drop = tf.layers.dropout(d0, dropout_rate, training=training)

        d1 = tf.layers.dense(
            d0_drop,
            200,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            activation=tf.nn.elu,
            name="d1"
        )
        d1_drop = tf.layers.dropout(d1, dropout_rate, training=training)

        d2 = tf.layers.dense(
            d1_drop,
            200,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            activation=tf.nn.elu,
            name="d2"
        )
        d2_drop = tf.layers.dropout(d2, dropout_rate, training=training)

        logits = tf.layers.dense(
            d2_drop,
            n_outputs,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.lambda_),
            # activation=tf.nn.elu,
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
        self._X, self._y = X, y
        self._training = training
        self._loss = loss
        self._y_prob = y_proba
        self._training_op = training_op
        self._init, self._saver = init, saver

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
