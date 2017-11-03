import numpy as np
from sklearn.preprocessing import normalize


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class CB(object):
    def __init__(self, ufd, ifd):
        self.ufd = ufd
        self.ifd = ifd

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def predict_proba(self, X):
        uid_m = normalize(self.ufd.m[X[:, 0]])
        iid_a_m = normalize(self.ifd.m[X[:, 1]])
        iid_b_m = normalize(self.ifd.m[X[:, 2]])

        a = iid_a_m.multiply(uid_m).sum(axis=1).A1
        b = iid_b_m.multiply(uid_m).sum(axis=1).A1
        logits = np.array([b - a, a - b])
        proba = softmax(logits)
        return proba[1]







