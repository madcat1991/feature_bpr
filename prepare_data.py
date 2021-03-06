"""
This script converts initial ratings data into training and testing sets.
"""

import argparse
import csv
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from data_tools.movielens import get_ratings_df
from data_tools.item_provider import get_item_feature_data, get_ifd_path
from data_tools.pairwise import get_training_path, get_testing_path
from data_tools.user_provider import get_user_feature_data, get_ufd_path

TEMPERATURE_COLD = 'cold'
TEMPERATURE_WARM = 'warm'
TEMPERATURES = [TEMPERATURE_COLD, TEMPERATURE_WARM]


def get_training_and_testing_dfs(rating_csv, split_q):
    df = get_ratings_df(rating_csv)
    logging.info("Ratings shape: %s", df.shape)

    border_ts = df.timestamp.quantile(q=split_q)
    training_df = df[df.timestamp <= border_ts].drop("timestamp", axis=1)
    testing_df = df[df.timestamp > border_ts].drop("timestamp", axis=1)
    logging.info("Training and testing shapes: %s, %s", training_df.shape, testing_df.shape)

    training_df = clean_bad_uids_from_df(training_df)
    testing_df = testing_df[testing_df.userId.isin(training_df.userId)]
    testing_df = clean_bad_uids_from_df(testing_df)

    logging.info("Training and testing after cleaning: %s, %s", training_df.shape, testing_df.shape)
    return training_df, testing_df


def clean_bad_uids_from_df(df):
    """The function leaves only users who rated two or more items with different ratings
    """
    logging.info("Before cleaning from bad uids: %s", df.shape)
    unique_ratings_per_user = df.groupby("userId").rating.nunique()
    good_users = unique_ratings_per_user[unique_ratings_per_user > 1].index
    df = df[df.userId.isin(good_users)]
    logging.info("After cleaning from bad uids: %s", df.shape)
    return df


def get_test_movie_obs(training_df, testing_df):
    if args.temperature == TEMPERATURE_COLD:
        return testing_df[~testing_df.movieId.isin(training_df.movieId)].movieId.value_counts()
    elif args.temperature == TEMPERATURE_WARM:
        return testing_df[testing_df.movieId.isin(training_df.movieId)].movieId.value_counts()
    else:
        return testing_df.movieId.value_counts()


def get_test_movie_sims(training_df, testing_df, ifd, step=1000):
    obs_per_tr_iids = training_df.movieId.value_counts()
    tr_m = normalize(ifd.get_objs_matrix(obs_per_tr_iids.index))

    if args.temperature == TEMPERATURE_COLD:
        ts_iids = testing_df[~testing_df.movieId.isin(obs_per_tr_iids.index)].movieId.unique()
    elif args.temperature == TEMPERATURE_WARM:
        ts_iids = testing_df[testing_df.movieId.isin(obs_per_tr_iids.index)].movieId.unique()
    else:
        ts_iids = testing_df.movieId.unique()

    logging.info("Searching similarity for each %s test items", ts_iids.size)
    ts_m = normalize(ifd.get_objs_matrix(ts_iids))

    sims = np.array([])
    for offset in range(0, ts_iids.size, step):
        limit = min(ts_iids.size, offset + step)
        sim_m = ts_m[offset: limit].dot(tr_m.T)
        sim_m = (sim_m.multiply(obs_per_tr_iids.values).sum(axis=1) / training_df.shape[0]).A1
        # sim_m = (sim_m.sum(axis=1) / tr_m.shape[0]).A1
        sims = np.r_[sims, sim_m]

    logging.info("Sim bins:\n%s", list(zip(*np.histogram(sims, bins=np.linspace(0, 1, 21)))))
    return pd.Series(data=sims, index=ts_iids)


def create_and_store_pairwise_data(df, output_path, movie_ids=None):
    logging.info("Converting DF with shape %s to pairwise preferences", df.shape)
    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "iid_a", "iid_b"])

        n_rows = 0
        for pw_data in iter_pairwise_data(df, movie_ids):
            writer.writerow(pw_data)
            n_rows += 1

    logging.info("Pairwise preferences [%s] have been stored to: %s", n_rows, output_path)


def iter_pairwise_data(df, movie_ids=None):
    for _, gp_df in df.groupby("userId"):
        uid = int(gp_df.iloc[0].userId)

        gp_df = gp_df.sample(frac=1)

        iids = gp_df.movieId.tolist()
        ratings = gp_df.rating.tolist()

        for i in range(1, len(iids)):
            iid1, rating1 = iids[i - 1], ratings[i - 1]
            iid2, rating2 = iids[i], ratings[i]

            if movie_ids is None or iid1 in movie_ids or iid2 in movie_ids:
                if rating1 > rating2:
                    yield uid, iid1, iid2
                elif rating1 < rating2:
                    yield uid, iid2, iid1


def main():
    logging.info("Start")

    rating_tr_df, rating_ts_df = get_training_and_testing_dfs(args.rating_csv, args.split_q)

    # preparing feature data
    ifd = get_item_feature_data(get_ifd_path(args.data_dir), args.movie_csv, args.tag_csv, args.rebuild_ifd)
    if args.rebuild_ufd:
        # we don't need it in other parts of the code
        get_user_feature_data(get_ufd_path(args.data_dir), rating_tr_df, ifd, args.rebuild_ufd)

    # preparing pairwise data
    create_and_store_pairwise_data(rating_tr_df, get_training_path(args.data_dir))
    if args.n_obs:
        obs_per_test_movie = get_test_movie_obs(rating_tr_df, rating_ts_df)
        for n_obs in args.n_obs:
            logging.info("Collecting testing data for movies with %s observations", n_obs)
            movie_ids = set(obs_per_test_movie[obs_per_test_movie == n_obs].index)
            path = get_testing_path(args.data_dir, args.temperature, n_obs=n_obs)
            create_and_store_pairwise_data(rating_ts_df, path, movie_ids)
    elif args.sims:
        sim_per_test_movie = get_test_movie_sims(rating_tr_df, rating_ts_df, ifd)
        for sim in args.sims:
            logging.info("Collecting testing data for movies with %.2f similarity", sim)

            idx = np.isclose(sim_per_test_movie.values, sim, atol=0.05)
            sims = sim_per_test_movie.values[idx]
            logging.info("Min sim: %.3f, max sim: %.3f", min(sims), max(sims))

            movie_ids = set(sim_per_test_movie.index[idx])
            path = get_testing_path(args.data_dir, args.temperature, sim=sim)
            create_and_store_pairwise_data(rating_ts_df, path, movie_ids)
    else:
        if args.temperature == TEMPERATURE_COLD:
            movie_ids = set(rating_ts_df[~rating_ts_df.movieId.isin(rating_tr_df.movieId)].movieId)
        elif args.temperature == TEMPERATURE_WARM:
            movie_ids = set(rating_ts_df[rating_ts_df.movieId.isin(rating_tr_df.movieId)].movieId)
        else:
            movie_ids = set(rating_ts_df.movieId)

        path = get_testing_path(args.data_dir, args.temperature)
        create_and_store_pairwise_data(rating_ts_df, path, movie_ids)

    logging.info("Stop")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-r', required=True, dest="rating_csv", help='Path to a csv file with ratings')
    parser.add_argument('-m', dest="movie_csv", help='Path to the csv file with movies')
    parser.add_argument('-g', dest="tag_csv", help='Path to the csv file with movie tags')
    parser.add_argument('--ifd', dest='rebuild_ifd', action='store_true',
                        help="Rebuild item-feature data")
    parser.add_argument('--ufd', dest='rebuild_ufd', action='store_true',
                        help="Rebuild user-feature data")

    parser.add_argument('-p', default="data", dest="data_dir",
                        help='Path to the data directory to save results. Default: data/')

    parser.add_argument('-q', default=0.8, type=float, dest="split_q",
                        help='Splitting ratings by timestamp quantile. Default: 0.8')

    parser.add_argument('-o', dest="n_obs", type=int, nargs='+',
                        help='The number of observations for cold-start item in the test set. '
                             'If specified, the testing data is generated based on observations')
    parser.add_argument('-s', dest="sims", type=float, nargs='+',
                        help='How similar the test items should be? '
                             'If specified, the testing data is generated based on similarities')
    parser.add_argument('-t', dest="temperature", default=None, choices=TEMPERATURES,
                        help='The type of test items. Default: any')

    parser.add_argument("--log-level", default='INFO', dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNINGS', 'ERROR'], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s', stream=sys.stdout, level=getattr(logging, args.log_level)
    )

    main()
