import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy_score_avg_by_users(y_true, y_pred, uids):
    uid_accuracies = []
    for uid in np.unique(uids):
        idx = np.where(uids == uid)[0]
        uid_accuracies.append(accuracy_score(y_true[idx], y_pred[idx]))
    return np.mean(uid_accuracies)


def roc_auc_score_avg_by_users(y_true, y_pred, uids):
    uid_aucs = []
    for uid in np.unique(uids):
        idx = np.where(uids == uid)[0]
        uid_aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.mean(uid_aucs)
