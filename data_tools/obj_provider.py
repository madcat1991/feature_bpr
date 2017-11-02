import logging
import pickle

from scipy.io import mmwrite, mmread


class ObjFeatureData(object):
    def __init__(self, m, obj_to_row, feature_to_col):
        self.m = m
        self.obj_to_row = obj_to_row
        self.feature_to_col = feature_to_col

    def info(self):
        return {
            "nnz": self.m.nnz,
            "density": self.m.nnz / (self.m.shape[0] * self.m.shape[1]),
            "shape": self.m.shape
        }

    def get_obj_vector(self, obj_id, dense=False):
        item_vec = self.m[self.obj_to_row[obj_id]]
        if dense:
            item_vec = item_vec.todense().reshape(-1)
        return item_vec

    def get_objs_matrix(self, objs_ids):
        row_ids = [self.obj_to_row[obj_id] for obj_id in objs_ids]
        return self.m[row_ids]

    @classmethod
    def create(cls, *args, **kwargs):
        return ObjFeatureData(None, {}, {})

    def save(self, pkl_path):
        logging.info("Dumping %s", self.__class__)
        data = {
            "obj_to_row": self.obj_to_row,
            "feature_to_col": self.feature_to_col,
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info("Dict data has been dumped")

        matrix_path = pkl_path + ".mtx"
        mmwrite(matrix_path, self.m)
        logging.info("Matrix has been dumped")

    @classmethod
    def load(cls, pkl_path):
        logging.info("Loading %s", cls.__class__)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        logging.info("Pickle data has been loaded")

        matrix_path = pkl_path + ".mtx"
        m = mmread(matrix_path).tocsr()
        logging.info("Matrix has been loaded")
        return cls(m, **data)
