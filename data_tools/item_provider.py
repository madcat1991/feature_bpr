import logging
import os

from scipy.sparse import csr_matrix

from data_tools.movielens import get_movies_df, get_tags_df
from data_tools.obj_provider import ObjFeatureData


class ItemFeatureData(ObjFeatureData):
    @classmethod
    def create(cls, movie_csv_path, tag_csv_path):
        logging.info("Creating %s", cls.__class__)

        iid_to_row = {}
        feature_to_col = {}

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
        return cls(m, iid_to_row, feature_to_col)


def get_item_feature_data(ifd_path, movie_csv=None, tag_csv=None):
    if os.path.exists(ifd_path):
        ifd = ItemFeatureData.load(ifd_path)
    elif movie_csv is None or tag_csv is None:
        raise Exception("There is no dumped item-feature data, please specify movie_csv and tag_csv")
    else:
        ifd = ItemFeatureData.create(movie_csv, tag_csv)
        ifd.save(ifd_path)
    logging.info("Item-feature data: %s", ifd.info())
    return ifd


def get_ifd_path(data_dir=None, name="ifd.pkl"):
    return os.path.join(data_dir, name) if data_dir else name
