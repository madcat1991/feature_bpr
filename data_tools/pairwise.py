import logging

import numpy as np


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
