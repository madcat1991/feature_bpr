import logging

import os
from scipy.sparse import vstack

from data_tools.obj_provider import ObjFeatureData


class UserFeatureData(ObjFeatureData):
    @classmethod
    def create(cls, ratings_df, ifd):
        logging.info("Creating %s", cls.__class__)
        uid_to_row = {}
        rows = []
        for uid in ratings_df.userId.unique():
            uid_df = ratings_df[ratings_df.userId == uid]
            uid_m = ifd.get_objs_matrix(uid_df.movieId.values)
            uid_m = uid_m.multiply(uid_df.rating.values)
            uid_row = uid_m.sum(axis=0) / uid_df.rating.sum()

            uid_to_row.setdefault(uid, len(uid_to_row))
            rows.append(uid_row)

        m = vstack(rows, 'csr')
        logging.info("Item-feature data has been created")
        return cls(m, uid_to_row, ifd.feature_to_col)


def get_user_feature_data(ufd_path, rating_df=None, ifd=None):
    if os.path.exists(ufd_path):
        ufd = UserFeatureData.load(ufd_path)
    elif rating_df is None or ifd is None:
        raise Exception("There is no dumped user-feature data, please specify rating_df and ifd")
    else:
        ufd = UserFeatureData.create(rating_df, ifd)
        ufd.save(ufd_path)
    logging.info("User-feature data: %s", ufd.info())
    return ufd


def get_ifd_path(data_dir=None, name="udf.pkl"):
    return os.path.join(data_dir, name) if data_dir else name
