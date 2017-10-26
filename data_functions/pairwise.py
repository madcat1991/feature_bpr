import pandas as pd


def normailze_uids_and_iids(df, uid_cat=None, iid_cat=None):
    if not uid_cat:
        uid_cat = df.uid.astype('category').cat

    if not iid_cat:
        iid_cat = pd.Categorical(df[["iid_a", "iid_b"]].values.reshape(-1))

    df.uid = df.uid.astype('category', categories=uid_cat.categories).cat.codes
    df.iid_a = df.iid_a.astype('category', categories=iid_cat.categories).cat.codes
    df.iid_b = df.iid_b.astype('category', categories=iid_cat.categories).cat.codes

    return df, uid_cat, iid_cat
