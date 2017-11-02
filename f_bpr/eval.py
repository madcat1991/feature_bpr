"""This script evaluates the fPW model
"""

import argparse
import logging
import sys

import numpy as np

from common import sample_negative, find_best_params
from data_tools.item_provider import get_item_feature_data
from data_tools.pairwise import load_data, get_training_path, get_testing_path
from f_bpr.model import FBPR
from metrics import accuracy_score_avg_by_users, bpr_auc_by_users


def main():
    ifd = get_item_feature_data(args.pkl_data, args.movie_csv, args.tag_csv)
    item_feature_m = ifd.m.todense()

    X, uid_idx, _ = load_data(get_training_path(args.data_dir), iid_idx=ifd.iid_to_row)
    X, y = sample_negative(X)

    param_grid = {
        "n_epochs": [5],
        "n_factors": [10],
        "lambda_p": [0.1],
        "lambda_w": [0.1],
        "learning_rate": [0.001],
        "random_state": [args.random_state],
        "batch_size": [50000],
        "n_users": [len(uid_idx)],
        "n_items": [len(ifd.iid_to_row)],
        "n_features": [len(ifd.feature_to_col)]
    }

    best_params = find_best_params(
        X, y, FBPR, param_grid, random_state=args.random_state,
        item_feature_m=item_feature_m
    )

    logging.info("Training final bpr, params: %s", best_params)
    bpr = FBPR(**best_params)
    bpr.fit(X, y, item_feature_m=item_feature_m)

    for temperature in ["warm", "cold", None]:
        testing_path = get_testing_path(args.data_dir, temperature)
        X_test, _, _ = load_data(testing_path, uid_idx, ifd.iid_to_row)
        X_test, y_test = sample_negative(X_test)

        y_proba = np.array([])
        y_pred = np.array([])
        for offset in range(0, X_test.shape[0], args.step):
            limit = min(offset + args.step, X_test.shape[0])
            X_test_step = X_test[offset: limit]

            y_proba = np.r_[y_proba, bpr.predict_proba(X_test_step)]
            y_pred = np.r_[y_pred, bpr.predict(X_test_step)]

        uids = X_test[:, 0].reshape(-1)
        auc = bpr_auc_by_users(y_test, y_proba, uids)
        acc = accuracy_score_avg_by_users(y_test, y_pred, uids)
        logging.info("Test: acc=%.3f, auc=%.3f", acc, auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p', default="data", dest="data_dir",
                        help='Path to the data directory. Default: data/')

    parser.add_argument('-d', dest="pkl_data", default="ifd.pkl",
                        help='Path to the *.pkl file with the item-feature data. Default: ifd.pkl')
    parser.add_argument('-m', dest="movie_csv", help='Path to the csv file with movies')
    parser.add_argument('-t', dest="tag_csv", help='Path to the csv file with movie tags')

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
