"""
This script evaluates the BPR model
"""

import argparse
import logging
import sys

import pandas as pd
from sklearn.model_selection import ParameterGrid, ShuffleSplit

from bpr.model import BPR
from data_tools.pairwise import normailze_uids_and_iids


def load_data(csv_path, uid_idx=None, iid_idx=None):
    logging.info("Loading pairwise data from: %s", csv_path)
    pw_df = pd.read_csv(csv_path)
    if uid_idx and iid_idx:
        pw_df, _, _ = normailze_uids_and_iids(pw_df, uid_idx, iid_idx)
    else:
        pw_df, uid_idx, iid_idx = normailze_uids_and_iids(pw_df)
    return pw_df.values, uid_idx, iid_idx


def main():
    logging.info("Loading training data")
    X, uid_idx, iid_idx = load_data(args.training_csv)

    n_users = len(uid_idx)
    n_items = len(iid_idx)

    param_grid = {
        "n_epochs": [5],
        "n_factors": [10],
        "lambda_": [0.01],
        "learning_rate": [0.01],
        "random_state": [args.random_state],
        "batch_size": [10000],
    }

    logging.info("Starting grid search")
    best_params = best_auc = None
    for params in ParameterGrid(param_grid):
        logging.info("Evaluating params: %s", params)
        bpr = BPR(n_users, n_items, **params)

        splitter = ShuffleSplit(n_splits=5, random_state=args.random_state)
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

    X_test, _, _ = load_data(args.testing_csv, uid_idx, iid_idx)

    auc = bpr.get_auc(X_test)
    logging.info("Test AUC: %.3f", auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tr', dest="training_csv", required=True,
                        help='Path to the pairwise training data')
    parser.add_argument('--ts', dest="testing_csv", required=True,
                        help='Path to the pairwise testing data')
    parser.add_argument('--rs', dest="random_state", type=int, default=42, help='Random state. Default: 42')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
