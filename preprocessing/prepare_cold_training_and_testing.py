"""
This script converts initial ratings data into training and testing sets.
The data is prepared for the evaluation of the cold item case: the testing
set contains only items which are not observed in the training set.
"""

import argparse
import logging
import sys

import numpy as np

from data_tools.movielens import get_ratings_df
from preprocessing.prepare_training_and_testing import clean_bad_uids_from_df, create_and_store_pairwise_data


def get_movie_ids(df):
    obs_per_movie = df.movieId.value_counts().sort_values()

    selection_size = args.n_items * 2
    if args.popularity == 'low':
        obs_per_movie = obs_per_movie.head(selection_size)
    elif args.popularity == 'high':
        obs_per_movie = obs_per_movie.tail(selection_size)

    logging.info("Observations per cold-start movie: %s", obs_per_movie.describe().to_dict())
    return np.random.choice(obs_per_movie.index, args.n_items, replace=False)


def get_training_and_testing_dfs():
    df = get_ratings_df(args.rating_csv)
    logging.info("Ratings shape: %s", df.shape)

    if args.split_q:
        border_ts = df.timestamp.quantile(q=args.split_q)
        training_df = df[df.timestamp <= border_ts].drop("timestamp", axis=1)
        testing_df = df[df.timestamp > border_ts].drop("timestamp", axis=1)
        logging.info("Training and testing shapes: %s, %s", training_df.shape, testing_df.shape)

        training_df = clean_bad_uids_from_df(training_df)
        testing_df = clean_bad_uids_from_df(testing_df)

        movie_ids = get_movie_ids(testing_df)

        training_df = training_df[~training_df.movieId.isin(movie_ids)]
        testing_df = testing_df[
            testing_df.userId.isin(training_df.userId) &
            testing_df.movieId.isin(movie_ids)
        ]
    else:
        df = get_ratings_df(args.rating_csv).drop("timestamp", axis=1)
        movie_ids = get_movie_ids(df)

        df = clean_bad_uids_from_df(df)

        training_df = df[~df.movieId.isin(movie_ids)]
        testing_df = df[
            df.userId.isin(training_df.userId) &
            df.movieId.isin(movie_ids)
        ]

    logging.info("Training and testing shapes before saving: %s, %s", training_df.shape, testing_df.shape)
    return training_df, testing_df


def main():
    logging.info("Start")
    training_df, testing_df = get_training_and_testing_dfs()
    create_and_store_pairwise_data(training_df, args.training_csv)
    create_and_store_pairwise_data(testing_df, args.testing_csv)
    logging.info("Stop")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', required=True, dest="rating_csv",
                        help='Path to a csv file with ratings')
    parser.add_argument('--tr', default="cold_training.csv", dest="training_csv",
                        help='Path to the training file. Default: cold_training.csv')
    parser.add_argument('--ts', default="cold_testing.csv", dest="testing_csv",
                        help='Path to the testing file. Default: cold_testing.csv')

    parser.add_argument('-q', default=None, type=float, dest="split_q",
                        help='Splitting ratings by timestamp quantile. If it is set, then '
                             'the data is split to the training and testing sets and the cold items'
                             'have only history observed in the testing set. If None, the cold items'
                             'use observations from the whole dataset. Default: None')
    parser.add_argument('-p', default='low', dest="popularity", choices=["low", "random", "high"],
                        help='The popularity of items selected for the cold start. '
                             'If -q is not None, then the popularity is calculated only within'
                             'the testing set. Default: low')
    parser.add_argument('-n', default=500, dest="n_items", type=int,
                        help='The number of test items. Default: 500')

    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
