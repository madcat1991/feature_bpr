import logging
import os

import numpy as np
from scipy.sparse import csr_matrix

from data_tools.obj_provider import ObjFeatureData


class UserFeatureData(ObjFeatureData):
    @property
    def n_users(self):
        return len(self.obj_to_row)

    @classmethod
    def create(cls, ratings_df, ifd):
        logging.info("Creating %s", cls)
        uid_to_row = {uid: row_id for row_id, uid in enumerate(ratings_df.userId.unique())}

        logging.info("Preparing user-item ratings matrix")
        rows = [uid_to_row[uid] for uid in ratings_df.userId.values]
        cols = [ifd.obj_to_row[iid] for iid in ratings_df.movieId.values]
        r_m = csr_matrix(
            (ratings_df.rating.values, (rows, cols)),
            (len(uid_to_row), ifd.n_items)
        )

        logging.info("Preparing user-feature matrix")
        m = None
        step = 1000
        for offset in range(0, r_m.shape[0], step):
            limit = min(offset + step, r_m.shape[0])
            _r_m_part = r_m[offset: limit]

            if m is None:
                m = _r_m_part.dot(ifd.m).todense()
            else:
                m = np.r_[m, _r_m_part.dot(ifd.m).todense()]

            if limit % 10000 == 0:
                logging.info("Processed %s/%s users", limit, r_m.shape[0])

        uid_ratings_sum = r_m.sum(axis=1).A1
        m = np.multiply(m, (1.0 / uid_ratings_sum).reshape(-1, 1))

        data = cls(m, uid_to_row, ifd.feature_to_col)
        logging.info("User-feature data has been created: %s", data.info())
        return data

    def info(self):
        return {"shape": self.m.shape}

    def get_obj_vector(self, obj_id, dense=False):
        item_vec = self.m[self.obj_to_row[obj_id]]
        if not dense:
            item_vec = csr_matrix(item_vec)
        return item_vec

    def _save_matrix(self, matrix_path):
        np.save(matrix_path, self.m)
        logging.info("Matrix has been dumped")

    @classmethod
    def _load_matrix(cls, matrix_path):
        m = np.load(matrix_path + '.npy')
        logging.info("Matrix has been dumped")
        return m


def get_user_feature_data(ufd_path, rating_df=None, ifd=None, rebuild=False):
    if os.path.exists(ufd_path) and not rebuild:
        ufd = UserFeatureData.load(ufd_path)
    elif rating_df is None or ifd is None:
        raise Exception("There is no dumped user-feature data, please specify rating_df and ifd")
    else:
        ufd = UserFeatureData.create(rating_df, ifd)
        ufd.save(ufd_path)
    logging.info("User-feature data: %s", ufd.info())
    return ufd


def get_ufd_path(data_dir=None, name="udf.pkl"):
    return os.path.join(data_dir, name) if data_dir else name
