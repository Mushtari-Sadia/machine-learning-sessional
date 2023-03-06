"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np


def accuracy(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    return np.mean(y_true == y_pred) * 100


def precision_score(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))

    return tp/(tp+fp)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return tp/(tp+fn)


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return (2*tp)/(2*tp+fp+fn)
