import logging

import pandas as pd
import sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, ShuffleSplit

from pw_data_functions import normailze_uids_and_iids, read_pw_data_as_X_and_y
from tf_bpr import TFBPR


def load_data(csv_path, uid_cat=None, iid_cat=None):
    pw_df = pd.read_csv(csv_path)
    if uid_cat and iid_cat:
        pw_df, _, _ = normailze_uids_and_iids(pw_df, uid_cat, iid_cat)
    else:
        pw_df, uid_cat, iid_cat = normailze_uids_and_iids(pw_df)
    X, y = read_pw_data_as_X_and_y(pw_df)
    return X, y, uid_cat, iid_cat


def main():
    logging.info("Loading training data")
    X, y, uid_cat, iid_cat = load_data("/Users/tural/PyProjects/feature_bpr/data/training.csv")

    n_users = uid_cat.categories.size
    n_items = iid_cat.categories.size
    random_state = 42
    n_epochs = 20

    param_grid = {
        "n_factors": [10],
        "lambda_": [0.01, 0.001],
        "learning_rate": [0.01, 0.001],
        "random_state": [random_state],
        "batch_size": [5000],
    }

    logging.info("Starting grid search")
    best_params = best_auc = None
    for params in ParameterGrid(param_grid):
        logging.info("Evaluating params: %s", params)
        bpr = TFBPR(n_users, n_items, **params)

        splitter = ShuffleSplit(n_splits=10, random_state=random_state)
        aucs = []
        for train_ids, test_ids in splitter.split(X, y):
            X_train, y_train = X[train_ids], y[train_ids]
            X_valid, y_valid = X[test_ids], y[test_ids]

            bpr.fit(X_train, y_train, n_epochs)

            y_pred = bpr.predict(X_valid)
            aucs.append(roc_auc_score(y_valid, y_pred))
            break

        auc = pd.np.mean(aucs)
        if best_auc is None or best_auc < auc:
            best_params = params
            logging.info("Validation AUC=%.3f, params: %s", auc, params)

    logging.info("Training final bpr, params: %s", best_params)
    bpr = TFBPR(n_users, n_items, **best_params)
    bpr.fit(X, y, n_epochs)

    logging.info("Loading testing data")
    X_test, y_test, _, _ = load_data("/Users/tural/PyProjects/feature_bpr/data/testing.csv", uid_cat, iid_cat)

    y_pred = bpr.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    logging.info("Test AUC: %.3f", auc)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=logging.INFO
    )

    main()
