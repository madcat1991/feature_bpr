"""
This script evaluates the PW model
"""

import argparse
import logging
import sys

import numpy as np

from common import find_best_params, sample_negative, load_data
from metrics import accuracy_score_avg_by_users, bpr_auc_by_users
from bpr.model import BPR


def main():
    X, uid_idx, iid_idx = load_data(args.training_csv)
    X, y = sample_negative(X)

    param_grid = {
        "n_epochs": [5],
        "n_factors": [5, 10, 20],
        "lambda_p": [0.01],
        "lambda_q": [0.01],
        "learning_rate": [0.01],
        "random_state": [args.random_state],
        "batch_size": [10000],
        "n_users": [len(uid_idx)],
        "n_items": [len(iid_idx)]
    }

    best_params = find_best_params(X, y, BPR, param_grid, args.test_size, random_state=args.random_state)

    logging.info("Training final bpr, params: %s", best_params)
    bpr = BPR(**best_params)
    bpr.fit(X, y)

    X_test, _, _ = load_data(args.testing_csv, uid_idx, iid_idx)
    X_test = X_test[np.random.choice(X_test.shape[0], args.test_size, replace=False)]
    X_test, y_test = sample_negative(X_test)

    uids = X_test[:, 0].reshape(-1)
    acc = accuracy_score_avg_by_users(y_test, bpr.predict(X_test), uids)
    auc = bpr_auc_by_users(y_test, bpr.predict_proba(X_test), uids)
    logging.info("Test: acc=%.3f, auc=%.3f", acc, auc)


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
