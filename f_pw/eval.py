"""This script evaluates the fPW model
"""

import argparse
import logging
import sys

import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, ShuffleSplit

from bpr.eval import load_data
from data_tools.provider import ItemFeatureData
from f_pw.model import FPWClassifier


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


def get_item_feature_data():
    if os.path.exists(args.pkl_data):
        ifd = ItemFeatureData.load(args.pkl_data)
    elif args.movie_csv is None or args.tag_csv is None:
        raise Exception("There is no dumped item-feature data, please specify movie_csv and tag_csv")
    else:
        ifd = ItemFeatureData.create(args.movie_csv, args.tag_csv)
        ifd.save(args.pkl_data)
    logging.info("Item-feature data: %s", ifd.info())
    return ifd


def main():
    ifd = get_item_feature_data()
    item_feature_m = ifd.m.todense()

    X, uid_idx, _ = load_data(args.training_csv, iid_idx=ifd.iid_to_row)
    X, y = sample_negative(X)

    n_users = len(uid_idx)
    n_items = len(ifd.iid_to_row)
    n_features = len(ifd.feature_to_col)

    param_grid = {
        "n_epochs": [5],
        "n_factors": [20],
        "lambda_": [0.01],
        "learning_rate": [0.01],
        "random_state": [args.random_state],
        "batch_size": [10000],
    }

    logging.info("Starting grid search")
    best_params = best_auc = None
    for params in ParameterGrid(param_grid):
        best_params = params
        break

        logging.info("Evaluating params: %s", params)
        bpr = FPWClassifier(n_users, n_items, n_features, **params)

        aucs = []
        splitter = ShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        for train_ids, valid_ids in splitter.split(X):
            bpr.fit(X[train_ids], y[train_ids], item_feature_m)
            aucs.append(roc_auc_score(y[valid_ids], bpr.predict(X[valid_ids])))

        auc = pd.np.mean(aucs)
        if best_auc is None or best_auc < auc:
            best_params = params
            best_auc = auc
            logging.info("Best AUC=%.3f, params: %s", auc, params)

    logging.info("Training final bpr, params: %s", best_params)
    bpr = FPWClassifier(n_users, n_items, n_features, **best_params)
    bpr.fit(X, y, item_feature_m)

    X_test, _, _ = load_data(args.testing_csv, uid_idx, ifd.iid_to_row)
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

    parser.add_argument('-d', dest="pkl_data", default="ifd.pkl",
                        help='Path to the *.pkl file with the item-feature data. Default: ifd.pkl')
    parser.add_argument('-m', dest="movie_csv", help='Path to the csv file with movies')
    parser.add_argument('-t', dest="tag_csv", help='Path to the csv file with movie tags')

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
