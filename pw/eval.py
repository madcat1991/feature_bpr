"""
This script evaluates the BPR model
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, ShuffleSplit

from bpr.eval import load_data
from f_pw.eval import sample_negative
from pw.model import PWClassifier


def main():
    logging.info("Loading training data")
    X, uid_idx, iid_idx = load_data(args.training_csv)
    X, y = sample_negative(X)

    n_users = len(uid_idx)
    n_items = len(iid_idx)

    param_grid = {
        "n_epochs": [3],
        "n_factors": [5],
        "lambda_": [0.01],
        "learning_rate": [0.01],
        "random_state": [args.random_state],
        "batch_size": [10000],
    }

    logging.info("Starting grid search")
    best_params = best_auc = None
    for params in ParameterGrid(param_grid):
        logging.info("Evaluating params: %s", params)
        bpr = PWClassifier(n_users, n_items, **params)

        splitter = ShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        aucs = []
        for train_ids, valid_ids in splitter.split(X):
            bpr.fit(X[train_ids], y[train_ids])
            aucs.append(roc_auc_score(y[valid_ids], bpr.predict(X[valid_ids])))

        auc = pd.np.mean(aucs)
        if best_auc is None or best_auc < auc:
            best_params = params
            best_auc = auc
            logging.info("Best AUC=%.3f, params: %s", auc, params)

    logging.info("Training final bpr, params: %s", best_params)
    bpr = PWClassifier(n_users, n_items, **best_params)
    bpr.fit(X, y)

    X_test, _, _ = load_data(args.testing_csv, uid_idx, iid_idx)
    X_test = X_test[np.random.choice(X_test.shape[0], args.test_size, replace=False)]
    X_test, y_test = sample_negative(X_test)

    auc = roc_auc_score(y_test, bpr.predict(X_test))
    logging.info("Test AUC: %.3f", auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tr', dest="training_csv", required=True,
                        help='Path to the pairwise training data')
    parser.add_argument('--ts', dest="testing_csv", required=True,
                        help='Path to the pairwise testing data')
    parser.add_argument('--rs', dest="random_state", type=int, default=42, help='Random state. Default: 42')

    parser.add_argument('-s', dest="test_size", type=int, default=40000,
                        help='The size of the test. Default: 40000')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()