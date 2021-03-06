"""
This script evaluates the DL-PW model
"""

import argparse
import logging
import sys

import numpy as np

from common import sample_negative, find_best_params
from data_tools.item_provider import get_item_feature_data, get_ifd_path
from data_tools.pairwise import load_data, get_training_path, get_testing_path
from dl_pw.model import DLPW
from metrics import accuracy_score_avg_by_users, bpr_auc_by_users


def main():
    ifd = get_item_feature_data(get_ifd_path(args.data_dir))
    item_feature_m = ifd.m.todense()

    X, uid_idx, iid_idx = load_data(get_training_path(args.data_dir), iid_idx=ifd.obj_to_row)
    X, y = sample_negative(X)

    param_grid = {
        "n_epochs": [5],
        "n_factors": [10],
        "lambda_ol": [0.1, 0.01, 0.001],
        "lambda_hl": [0.1, 0.01, 0.001],
        "h_layers": [[32, 32, 32]],
        "learning_rate": [0.01, 0.001],
        "dropout_rate": [None, 0.25, 0.5],
        "batch_norm_momentum": [None, 0.95, 0.99],
        "random_state": [args.random_state],
        "batch_size": [50000],
        "n_users": [len(uid_idx)],
        "n_items": [ifd.n_items],
        "n_features": [ifd.n_features]
    }

    best_params = find_best_params(
        X, y, DLPW, param_grid, random_state=args.random_state,
        item_feature_m=item_feature_m
    )

    logging.info("Training final fpw, params: %s", best_params)
    pw = DLPW(**best_params)
    pw.fit(X, y, item_feature_m=item_feature_m)

    for temperature in ["warm", "cold", None]:
        testing_path = get_testing_path(args.data_dir, temperature)
        X_test, _, _ = load_data(testing_path, uid_idx, iid_idx)
        X_test, y_test = sample_negative(X_test)

        y_proba = np.array([])
        y_pred = np.array([])
        for offset in range(0, X_test.shape[0], args.step):
            limit = min(offset + args.step, X_test.shape[0])
            X_test_step = X_test[offset: limit]

            y_proba = np.r_[y_proba, pw.predict_proba(X_test_step)]
            y_pred = np.r_[y_pred, pw.predict(X_test_step)]

        uids = X_test[:, 0].reshape(-1)
        auc = bpr_auc_by_users(y_test, y_proba, uids)
        acc = accuracy_score_avg_by_users(y_test, y_pred, uids)
        logging.info("Test: acc=%.3f, auc=%.3f", acc, auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', default="data", dest="data_dir",
                        help='Path to the data directory. Default: data/')
    parser.add_argument('--rs', dest="random_state", type=int, default=42, help='Random state. Default: 42')
    parser.add_argument('-s', dest="step", type=int, default=40000,
                        help='The size of the test step. Default: 40000')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
