import logging
import os
import pickle

from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix

from data_tools.movielens import get_movies_df, get_tags_df


class ItemFeatureData(object):
    def __init__(self, m, iid_to_row, feature_to_col):
        self.m = m
        self.iid_to_row = iid_to_row
        self.feature_to_col = feature_to_col

    def info(self):
        return {
            "nnz": self.m.nnz,
            "density": self.m.nnz / (self.m.shape[0] * self.m.shape[1]),
            "shape": self.m.shape
        }

    def get_item_vector(self, item_id, dense=False):
        item_vec = self.m[self.iid_to_row[item_id]]
        if dense:
            item_vec = item_vec.todense().reshape(-1)
        return item_vec

    @staticmethod
    def create(movie_csv_path, tag_csv_path):
        logging.info("Creating item-feature data")
        feature_to_col = {}
        iid_to_row = {}

        cols = []
        rows = []
        data = []

        logging.info("Collecting data about genres")
        df = get_movies_df(movie_csv_path)
        for t in df.itertuples():
            genres = t.genres.split("|")
            if genres:
                row_id = iid_to_row.setdefault(t.movieId, len(iid_to_row))
                for g in genres:
                    col_id = feature_to_col.setdefault(g, len(feature_to_col))

                    rows.append(row_id)
                    cols.append(col_id)
                    data.append(1)

        logging.info("Collecting data about tags")
        df = get_tags_df(tag_csv_path)
        for t in df.itertuples():
            row_id = iid_to_row.setdefault(t.movieId, len(iid_to_row))
            col_id = feature_to_col.setdefault("f%s" % t.tagId, len(feature_to_col))

            rows.append(row_id)
            cols.append(col_id)
            data.append(float(t.relevance))

        m = csr_matrix((data, (rows, cols)))
        logging.info("Item-feature data has been created")
        return ItemFeatureData(m, iid_to_row, feature_to_col)

    def save(self, pkl_path):
        logging.info("Dumping item-feature data")
        data = {
            "iid_to_row": self.iid_to_row,
            "feature_to_col": self.feature_to_col,
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info("Pickle data has been dumped")

        matrix_path = pkl_path + ".mtx"
        mmwrite(matrix_path, self.m)
        logging.info("Item-feature matrix has been dumped")

    @staticmethod
    def load(pkl_path):
        logging.info("Loading item-feature data")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        logging.info("Pickle data has been loaded")

        matrix_path = pkl_path + ".mtx"
        m = mmread(matrix_path).tocsr()
        logging.info("Item-feature matrix has been loaded")
        return ItemFeatureData(m, **data)


def get_item_feature_data(pkl_data, movie_csv=None, tag_csv=None):
    if os.path.exists(pkl_data):
        ifd = ItemFeatureData.load(pkl_data)
    elif movie_csv is None or tag_csv is None:
        raise Exception("There is no dumped item-feature data, please specify movie_csv and tag_csv")
    else:
        ifd = ItemFeatureData.create(movie_csv, tag_csv)
        ifd.save(pkl_data)
    logging.info("Item-feature data: %s", ifd.info())
    return ifd
