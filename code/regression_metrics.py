import numpy as np

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def gini_coefficient(x):
    # half of relative mean difference
    sorted_x = sorted(x)
    len_x = len(x)
    tot = 0.0
    for i, xi in enumerate(sorted_x):
        added = i
        subtracted = len_x - i - 1
        tot += (added - subtracted) * xi
    return tot / (len_x ** 2)


def max_error(y_true, pred):
    return max(np.abs(y_true - pred))


def error_variance(y_true, pred):
    return np.std(y_true - pred) ** 2


def relative_error_variance(y_true, pred):
    return (np.std(y_true - pred) / np.std(y_true)) ** 2


def gini_loss(y_true, pred):
    return gini_coefficient(y_true - pred)

REGRESSION_METRICS = (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    error_variance,
    relative_error_variance,
    gini_loss,
)
