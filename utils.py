import numpy as np
import math

from sklearn import metrics as sk_metrics

def metrics(y, pred, name):
    ###########################
    #####Performance Metrics###
    ###########################

    if name.lower() == 'mse':
        return mse(pred, y)

    elif name.lower() == 'auc':
        # def auc(pred, y):

        #     n = 0
        #     correct = 0

        #     for i in range(len(y)):
        #        for j in range(i+1, len(y)):
        #             if (pred[i] - pred[j]) * (y[i] - y[j]) > 0:
        #                 correct += 1

        #             n += 1

        #     return correct * 1.0 / n
        return sk_metrics.roc_auc_score(y, pred)
    elif name.lower() == "accuracy":

        if len(np.unique(pred)) > 10:
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
        elif len(np.unique(y)) > 10:
            y[y > 0.5] = 1
            y[y <= 0.5] = 1

        return np.mean(y == pred)

    elif name.lower() == "logistic":
        return sk_metrics.log_loss(y, pred)
