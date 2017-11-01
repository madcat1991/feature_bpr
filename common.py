import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ShuffleSplit

from data_tools.pairwise import normailze_uids_and_iids
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
            estimator.fit(X[train_ids], y[train_ids], **fit_kwargs)

            X_valid, y_valid = X[valid_ids], y[valid_ids]
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


def load_data(csv_path, uid_idx=None, iid_idx=None):
    logging.info("Loading pairwise data from: %s", csv_path)
    pw_df = pd.read_csv(csv_path)
    if uid_idx and iid_idx:
        pw_df, _, _ = normailze_uids_and_iids(pw_df, uid_idx, iid_idx)
    else:
        pw_df, uid_idx, iid_idx = normailze_uids_and_iids(pw_df)
    return pw_df.values, uid_idx, iid_idx


def get_training_path(data_dir=None):
    name = "training.csv"
    return os.path.join(data_dir, name) if data_dir else name


def get_testing_path(is_cold, n_obs=None, sim=None, data_dir=None):
    name = "testing"
    if is_cold:
        name += "_cold"
    if n_obs is not None:
        name += "_obs_%d" % n_obs
    elif sim is not None:
        name += "_sim_%d" % int(sim * 100)
    name = "%s.csv" % name
    return os.path.join(data_dir, name) if data_dir else name
