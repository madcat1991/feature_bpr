"""This script evaluates the fPW model
"""

import argparse
import logging
import sys

import numpy as np

from cb.model import CB
from common import sample_negative
from data_tools.item_provider import get_item_feature_data, get_ifd_path
from data_tools.pairwise import load_data, get_testing_path
from data_tools.user_provider import get_user_feature_data, get_ufd_path
from metrics import accuracy_score_avg_by_users, bpr_auc_by_users


def main():
    ifd = get_item_feature_data(get_ifd_path(args.data_dir))
    ufd = get_user_feature_data(get_ufd_path(args.data_dir))
    cb = CB(ufd, ifd)

    for temperature in ["warm", "cold", None]:
        testing_path = get_testing_path(args.data_dir, temperature)
        X_test, _, _ = load_data(testing_path, ufd.obj_to_row, ifd.obj_to_row)
        X_test, y_test = sample_negative(X_test)

        y_proba = np.array([])
        y_pred = np.array([])
        for offset in range(0, X_test.shape[0], args.step):
            limit = min(offset + args.step, X_test.shape[0])
            X_test_step = X_test[offset: limit]

            y_proba = np.r_[y_proba, cb.predict_proba(X_test_step)]
            y_pred = np.r_[y_pred, cb.predict(X_test_step)]

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
