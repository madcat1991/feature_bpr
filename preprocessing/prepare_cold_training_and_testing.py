"""
This script converts initial ratings data into training and testing sets,
splitting them by time. The data is prepared to evaluate the cold item
case
"""
import argparse
import logging
import sys

import numpy as np

from data_tools.movielens import get_ratings_df
from preprocessing.prepare_training_and_testing import clean_df, create_and_store_pairwise_data


def get_training_and_testing_dfs():
    df = get_ratings_df(args.rating_csv)
    logging.info("Ratings shape: %s", df.shape)

    border_ts = df.timestamp.quantile(q=args.split_q)
    training_df = df[df.timestamp <= border_ts].drop("timestamp", axis=1)
    testing_df = df[df.timestamp > border_ts].drop("timestamp", axis=1)
    logging.info("Training and testing shapes: %s, %s", training_df.shape, testing_df.shape)

    training_df = clean_df(training_df)
    testing_df = clean_df(testing_df)

    # we have 12387 items in training_df
    movie_ids = training_df.movieId.value_counts().head(5000)
    movie_ids = np.random.choice(movie_ids.index, 2000, replace=False)

    training_df = training_df[~training_df.movieId.isin(movie_ids)]
    testing_df = testing_df[
        testing_df.userId.isin(training_df.userId) &
        testing_df.movieId.isin(movie_ids)
    ]

    logging.info("Training and testing shapes after cleaning: %s, %s", training_df.shape, testing_df.shape)
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
    parser.add_argument('-q', default=0.8, type=float, dest="split_q",
                        help='Splitting timestamp quantile. Default: 0.8')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
