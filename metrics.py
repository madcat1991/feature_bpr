import numpy as np
from sklearn.metrics import accuracy_score


def accuracy_score_avg_by_users(y_true, y_pred, uids):
    acc_per_uid = []
    for uid in np.unique(uids):
        idx = np.where(uids == uid)[0]
        acc_per_uid.append(accuracy_score(y_true[idx], y_pred[idx]))
    return np.mean(acc_per_uid)


def bpr_auc_by_users(y_true, y_proba, uids):
    auc_per_uid = []
    true_proba = np.abs(y_true - 1.0 + y_proba)
    for uid in np.unique(uids):
        idx = np.where(uids == uid)[0]
        auc_per_uid.append(np.mean(true_proba[idx]))
    return np.mean(auc_per_uid)
