import logging
import os

import numpy as np
import pandas as pd


def normailze_uids_and_iids(df, uid_idx=None, iid_idx=None):
    if uid_idx is None:
        uids = df.uid.unique()
        uid_idx = {uid: _id for _id, uid in enumerate(uids)}

    if iid_idx is None:
        iids = np.unique(df[["iid_a", "iid_b"]].values.reshape(-1))
        iid_idx = {iid: _id for _id, iid in enumerate(iids)}

    df.uid = df.uid.map(uid_idx)
    df.iid_a = df.iid_a.map(iid_idx)
    df.iid_b = df.iid_b.map(iid_idx)

    # uid_idx and iid_idx are usually generated based on the training data
    # it is possible, that the test data will contain items which aren't
    # present in the training pairwise data(!) because items with equal ratings
    # aren't added to pairwise data sets
    logging.info("Shape after normalization: %s", df.shape)
    df = df.dropna()
    logging.info("Shape after dropping na: %s", df.shape)
    return df, uid_idx, iid_idx


def get_training_path(data_dir=None, name="training.csv"):
    return os.path.join(data_dir, name) if data_dir else name


def get_testing_path(data_dir=None, temperature=None, n_obs=None, sim=None):
    name = "testing"
    if temperature:
        name += "_%s" % temperature
    if n_obs is not None:
        name += "_obs_%d" % n_obs
    elif sim is not None:
        name += "_sim_%d" % int(sim * 100)
    name = "%s.csv" % name
    return os.path.join(data_dir, name) if data_dir else name


def load_data(csv_path, uid_idx=None, iid_idx=None):
    logging.info("Loading pairwise data from: %s", csv_path)
    pw_df = pd.read_csv(csv_path)
    if uid_idx and iid_idx:
        pw_df, _, _ = normailze_uids_and_iids(pw_df, uid_idx, iid_idx)
    else:
        pw_df, uid_idx, iid_idx = normailze_uids_and_iids(pw_df)
    return pw_df.values, uid_idx, iid_idx
