"""
This script converts initial ratings data into training and testing sets.
The data is prepared for the evaluation of the cold item case: the testing
set contains only items which are not observed in the training set.
"""

import argparse
import csv
import logging
import sys

import numpy as np

from data_tools.movielens import get_ratings_df
from train_and_test.plain import clean_bad_uids_from_df, create_and_store_pairwise_data


def get_movie_ids(df):
    obs_per_movie = df.movieId.value_counts()
    obs_per_movie = obs_per_movie[obs_per_movie == args.n_obs]

    if obs_per_movie.shape[0] > args.n_items:
        idx = np.random.choice(obs_per_movie.shape[0], args.n_items, False)
        obs_per_movie = obs_per_movie.iloc[idx]
    else:
        logging.info("The number of candidates [%s] < n_items. Taking them all")
    return set(obs_per_movie.index)


def iter_test_pairwise_data(df, movie_ids):
    for _, gp_df in df.groupby("userId"):
        uid = int(gp_df.iloc[0].userId)

        gp_df = gp_df.sample(frac=1)

        iids = gp_df.movieId.tolist()
        ratings = gp_df.rating.tolist()

        for i in range(1, len(iids)):
            iid1, rating1 = iids[i - 1], ratings[i - 1]
            iid2, rating2 = iids[i], ratings[i]

            if iid1 in movie_ids or iid2 in movie_ids:
                if rating1 > rating2:
                    yield uid, iid1, iid2
                elif rating1 < rating2:
                    yield uid, iid2, iid1


def create_and_store_test_pairwise_data(df, movie_ids, output_path):
    logging.info("Converting test DF with shape %s to pairwise preferences", df.shape)
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "iid_a", "iid_b"])

        n_rows = 0
        for pw_data in iter_test_pairwise_data(df, movie_ids):
            writer.writerow(pw_data)
            n_rows += 1

    logging.info("Test pairwise preferences [%s] have been stored to: %s", n_rows, output_path)


def get_training_and_testing_dfs():
    df = get_ratings_df(args.rating_csv)
    logging.info("Ratings shape: %s", df.shape)

    border_ts = df.timestamp.quantile(q=args.split_q)
    training_df = df[df.timestamp <= border_ts].drop("timestamp", axis=1)
    testing_df = df[df.timestamp > border_ts].drop("timestamp", axis=1)
    logging.info("Training and testing shapes: %s, %s", training_df.shape, testing_df.shape)

    training_df = clean_bad_uids_from_df(training_df)
    testing_df = clean_bad_uids_from_df(testing_df)

    movie_ids = get_movie_ids(testing_df)

    training_df = training_df[~training_df.movieId.isin(movie_ids)]
    testing_df = testing_df[testing_df.userId.isin(training_df.userId)]

    logging.info("Training and testing shapes before saving: %s, %s", training_df.shape, testing_df.shape)
    return training_df, testing_df, movie_ids


def main():
    logging.info("Start")
    training_df, testing_df, movie_ids = get_training_and_testing_dfs()
    create_and_store_pairwise_data(training_df, args.training_csv)
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

    parser.add_argument('-q', default=0.8, type=float, dest="split_q",
                        help='Splitting ratings by timestamp quantile. Default: 0.8')
    parser.add_argument('-s', default=0.1, dest="average_sim", type=float,
                        help='Average weighted feature similarity of testing and training items. '
                             'Default: 0.1')
    parser.add_argument('-n', default=1000, dest="n_items", type=int,
                        help='The number of cold-start items in the test set. Default: 1000')

    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
