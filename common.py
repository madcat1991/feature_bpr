import logging

import numpy as np
from sklearn.model_selection import ParameterGrid, ShuffleSplit

from metrics import bpr_auc_by_users


def find_best_params(X, y, est_class, param_grid, val_size=40000, n_splits=1, random_state=42, **fit_kwargs):
    logging.info("Starting grid search")
    best_params = best_auc = None
    for params in ParameterGrid(param_grid):
        logging.info("Evaluating params: %s", params)
        estimator = est_class(**params)

        splitter = ShuffleSplit(n_splits, val_size, random_state=random_state)
        aucs = []
        for train_ids, valid_ids in splitter.split(X):
            X_valid, y_valid = X[valid_ids], y[valid_ids]
            estimator.fit(X[train_ids], y[train_ids], X_valid, y_valid, **fit_kwargs)

            aucs.append(
                bpr_auc_by_users(y_valid, estimator.predict_proba(X_valid), X_valid[:, 0].reshape(-1))
            )

        auc = np.mean(aucs)
        if best_auc is None or best_auc < auc:
            best_params = params
            best_auc = auc
            logging.info("Best auc=%.3f, params: %s", auc, params)
    return best_params


def sample_negative(X, neg_p=0.5):
    size = int(X.shape[0] * neg_p)
    idx = np.random.choice(X.shape[0], size, False)

    # swapping negatives
    neg_X = X[idx]
    neg_X[:, [1, 2]] = neg_X[:, [2, 1]]
    X[idx] = neg_X

    y = np.ones(X.shape[0])
    y[idx] = 0
    return X, y
