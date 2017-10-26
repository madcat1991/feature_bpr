"""
This script converts initial ratings data into training and testing sets,
splitting them by time
"""
import argparse
import csv
import logging
import sys

from data_functions.movielens import get_ratings_df


def clean_df(df):
    """The function leaves only users who rated two or more items with different ratings
    """
    unique_ratings_per_user = df.groupby("userId").rating.nunique()
    good_users = unique_ratings_per_user[unique_ratings_per_user > 1].index
    df = df[df.userId.isin(good_users)]
    return df


def get_training_and_testing_dfs():
    df = get_ratings_df(args.rating_csv)
    logging.info("Ratings shape: %s", df.shape)

    border_ts = df.timestamp.quantile(q=args.split_q)
    training_df = df[df.timestamp <= border_ts].drop("timestamp", axis=1)
    testing_df = df[df.timestamp > border_ts].drop("timestamp", axis=1)
    logging.info("Training and testing shapes: %s, %s", training_df.shape, testing_df.shape)

    training_df = clean_df(training_df)
    testing_df = clean_df(testing_df)

    # removing users to whom we can't recommend anything
    testing_df = testing_df[testing_df.userId.isin(training_df.userId)]
    # removing items about which we don't have training data (for simple BPR)
    testing_df = testing_df[testing_df.movieId.isin(training_df.movieId)]

    logging.info("Training and testing shapes after cleaning: %s, %s", training_df.shape, testing_df.shape)
    return training_df, testing_df


def iter_pairwise_data(df):
    """
    Generating a list of pairwise preferences based on the partial information
    about the ratings. The direction is only a > b (a prefers b)

    Paper: "Personalized Ranking Recommendation via Integrating Multiple Feedbacks"

    :param df: input ratings data frame
    :return: (uid, a, b) tuples
    """
    for _, gp_df in df.groupby("userId"):
        uid = int(gp_df.iloc[0].userId)

        gp_df = gp_df.sample(frac=1)

        iids = gp_df.movieId.tolist()
        ratings = gp_df.rating.tolist()
        pairs = list(zip(iids, ratings))

        for _data in zip(pairs, pairs[1:]):
            (iid1, rating1), (iid2, rating2) = _data

            if rating1 > rating2:
                yield uid, iid1, iid2
            elif rating1 < rating2:
                yield uid, iid2, iid1


def create_and_store_pairwise_data(df, output_path):
    logging.info("Converting DF with shape %s to pairwise preferences", df.shape)
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "iid_a", "iid_b"])

        for pw_data in iter_pairwise_data(df):
            writer.writerow(pw_data)
    logging.info("Pairwise preferences have been stored to: %s", output_path)


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
    parser.add_argument('--tr', default="training.csv", dest="training_csv",
                        help='Path to the training file. Default: training.csv')
    parser.add_argument('--ts', default="testing.csv", dest="testing_csv",
                        help='Path to the testing file. Default: testing.csv')
    parser.add_argument('-q', default=0.8, type=float, dest="split_q",
                        help='Splitting timestamp quantile. Default: 0.8')
    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
