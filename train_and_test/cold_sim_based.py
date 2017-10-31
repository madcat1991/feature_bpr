"""
This script converts initial ratings data into training and testing sets.
The data is prepared for the evaluation of the cold item case: the testing
set contains only items which are not observed in the training set.
"""

import argparse
import logging
import sys

import numpy as np
from sklearn.preprocessing import normalize

from data_tools.provider import get_item_feature_data
from train_and_test.cold_obs_based import create_and_store_test_pairwise_data, get_training_and_testing_dfs
from train_and_test.plain import create_and_store_pairwise_data


def get_movie_ids(training_df, testing_df):
    ifd = get_item_feature_data(args.pkl_data)

    obs_per_tr_iids = training_df.movieId.value_counts()
    tr_m = normalize(ifd.get_items_matrix(obs_per_tr_iids.index))

    # only the movies which are not present in the training set
    ts_iids = testing_df[~testing_df.movieId.isin(obs_per_tr_iids.index)].movieId.unique()
    ts_m = normalize(ifd.get_items_matrix(ts_iids))

    sim_m = ts_m.dot(tr_m.T)
    sim_m = (sim_m.multiply(obs_per_tr_iids.values).sum(axis=1) / training_df.shape[0]).A1
    logging.info(
        "Sim bins:\n%s", list(zip(*np.histogram(sim_m, bins=np.linspace(0, 1, 21))))
    )

    diff = np.isclose(sim_m, args.average_sim, atol=0.05)
    idx = np.where(diff)[0]
    logging.info("Min sim: %.3f, max sim: %.3f", min(sim_m[idx]), max(sim_m[idx]))

    if idx.size > args.n_items:
        idx = np.random.choice(idx, args.n_items, False)
    else:
        logging.info("The number of candidates [%s] < n_items. Taking them all", idx.size)
    return set(ts_iids[idx])


def main():
    logging.info("Start")

    training_df, testing_df = get_training_and_testing_dfs(args.rating_csv, args.split_q)

    create_and_store_pairwise_data(training_df, args.training_csv)

    movie_ids = get_movie_ids(training_df, testing_df)
    create_and_store_test_pairwise_data(testing_df, movie_ids, args.testing_csv)

    logging.info("Stop")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', required=True, dest="rating_csv",
                        help='Path to a csv file with ratings')
    parser.add_argument('--tr', default="cold_training.csv", dest="training_csv",
                        help='Path to the training file. Default: cold_training.csv')
    parser.add_argument('--ts', default="cold_testing.csv", dest="testing_csv",
                        help='Path to the testing file. Default: cold_testing.csv')
    parser.add_argument('-d', dest="pkl_data", default="ifd.pkl",
                        help='Path to the *.pkl file with the item-feature data. Default: ifd.pkl')

    parser.add_argument('-q', default=0.8, type=float, dest="split_q",
                        help='Splitting ratings by timestamp quantile. Default: 0.8')
    parser.add_argument('-s', default=0.2, dest="average_sim", type=float,
                        help='Average weighted feature similarity of testing and training items. '
                             'Default: 0.2')
    parser.add_argument('-n', default=1000, dest="n_items", type=int,
                        help='The number of cold-start items in the test set. Default: 1000')

    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
