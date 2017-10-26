import numpy as np
import pandas as pd


UID_COL = 0
IID_A_COL = 1
IID_B_COL = 2


def normailze_uids_and_iids(df, uid_cat=None, iid_cat=None):
    if not uid_cat:
        uid_cat = df.uid.astype('category').cat

    if not iid_cat:
        iid_cat = pd.Categorical(df[["iid_a", "iid_b"]].values.reshape(-1))

    df.uid = df.uid.astype('category', categories=uid_cat.categories).cat.codes
    df.iid_a = df.iid_a.astype('category', categories=iid_cat.categories).cat.codes
    df.iid_b = df.iid_b.astype('category', categories=iid_cat.categories).cat.codes

    return df, uid_cat, iid_cat


def read_pw_data_as_X_and_y(pw_df, random_state=42, sample_negative=True, negative_p=0.5):
    y = np.ones(pw_df.shape[0])

    if sample_negative:
        pw_df = pw_df.sample(frac=1, random_state=random_state)
        n_neg = int(pw_df.shape[0] * negative_p)
        neg_ids = np.random.choice(pw_df.shape[0], n_neg, False)
        pw_df.iloc[neg_ids, [IID_A_COL, IID_B_COL]] = pw_df.iloc[neg_ids, [IID_B_COL, IID_A_COL]].values
        y[neg_ids] = 0

    return pw_df.values, y
