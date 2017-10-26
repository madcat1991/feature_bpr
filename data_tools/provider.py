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
    def load(movie_csv_path, tag_csv_path):
        feature_to_col = {}
        iid_to_row = {}

        cols = []
        rows = []
        data = []

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

        df = get_tags_df(tag_csv_path)
        for t in df.itertuples():
            row_id = iid_to_row.setdefault(t.movieId, len(iid_to_row))
            col_id = feature_to_col.setdefault("f%s" % t.tagId, len(feature_to_col))

            rows.append(row_id)
            cols.append(col_id)
            data.append(float(t.relevance))

        m = csr_matrix((data, (rows, cols)))
        return ItemFeatureData(m, iid_to_row, feature_to_col)
