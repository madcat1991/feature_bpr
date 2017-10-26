"""This script evaluates the BPR model
"""

import argparse
import logging
import sys

import pandas as pd
from sklearn.model_selection import ParameterGrid, ShuffleSplit

from bpr.model import BPR
from data_functions.pairwise import normailze_uids_and_iids


def load_data(csv_path, uid_cat=None, iid_cat=None):
    pw_df = pd.read_csv(csv_path)
    if uid_cat and iid_cat:
        pw_df, _, _ = normailze_uids_and_iids(pw_df, uid_cat, iid_cat)
    else:
        pw_df, uid_cat, iid_cat = normailze_uids_and_iids(pw_df)
    return pw_df.values, uid_cat, iid_cat


def main():
    logging.info("Loading training data")
    X, uid_cat, iid_cat = load_data(args.training_csv)

    n_users = uid_cat.categories.size
    n_items = iid_cat.categories.size
    random_state = 42

    param_grid = {
        "n_epochs": [10],
        "n_factors": [50],
        "lambda_": [0.001],
        "learning_rate": [0.01],
        "random_state": [random_state],
        "batch_size": [10000],
    }

    logging.info("Starting grid search")
    best_params = best_auc = None
    for params in ParameterGrid(param_grid):
        logging.info("Evaluating params: %s", params)
        bpr = BPR(n_users, n_items, **params)

        splitter = ShuffleSplit(n_splits=5, random_state=random_state)
        aucs = []
        for train_ids, valid_ids in splitter.split(X):
            bpr.fit(X[train_ids])
            aucs.append(bpr.get_auc(X[valid_ids]))
            break

        auc = pd.np.mean(aucs)
        if best_auc is None or best_auc < auc:
            best_params = params
            best_auc = auc
            logging.info("Best AUC=%.3f, params: %s", auc, params)

    logging.info("Training final bpr, params: %s", best_params)
    bpr = BPR(n_users, n_items, **best_params)
    bpr.fit(X)

    logging.info("Loading testing data")
    X_test, _, _ = load_data(args.testing_csv, uid_cat, iid_cat)
    auc = bpr.get_auc(X_test)

    logging.info("Test AUC: %.3f", auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tr', dest="training_csv", required=True,
                        help='Path to the pairwise training data')
    parser.add_argument('--ts', dest="testing_csv", required=True,
                        help='Path to the pairwise testing data')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
