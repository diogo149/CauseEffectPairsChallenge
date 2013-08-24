import numpy as np
from scipy.special import psi
from scipy.stats import skew, kurtosis, shapiro


def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    len_x = len(x)

    hx = 0.0
    for i in range(len_x - 1):
        delta = x[i + 1] - x[i]
        if delta != 0:
            hx += np.log(np.abs(delta))
    hx = hx / (len_x - 1) + psi(len_x) - psi(1)

    return hx


def num_unique(x):
    return len(set(x))


ALL_UNARY_FEATURES = (
)

NN_UNARY_FEATURES = (
    normalized_entropy,
    skew,
    kurtosis,
    np.std,
    shapiro,
)

CN_UNARY_FEATURES = (
)

CC_UNARY_FEATURES = (
    num_unique,
)

NC_UNARY_FEATURES = (
)


def unary_feature_wrapper(f):
    def inner_func(x, y):
        result = f(x)
        try:
            list_result = list(result)
        except TypeError:
            list_result = [result]
        feature_names = ["{}_{}".format(f.func_name, i) for i in range(len(list_result))]
        features = zip(feature_names, list_result)
        return features
    return inner_func

UNARY_FEATURES = []

for f in ALL_UNARY_FEATURES:
    for desired_type in ["NN", "NC", "CN", "CC"]:
        name = "{}_{}".format(desired_type, f.func_name)
        UNARY_FEATURES.append((name, unary_feature_wrapper, f))

for f in NN_UNARY_FEATURES:
    desired_type = "NN"
    name = "{}_{}".format(desired_type, f.func_name)
    UNARY_FEATURES.append((name, unary_feature_wrapper, f))

for f in CN_UNARY_FEATURES:
    desired_type = "CN"
    name = "{}_{}".format(desired_type, f.func_name)
    UNARY_FEATURES.append((name, unary_feature_wrapper, f))

for f in CC_UNARY_FEATURES:
    desired_type = "CC"
    name = "{}_{}".format(desired_type, f.func_name)
    UNARY_FEATURES.append((name, unary_feature_wrapper, f))

for f in NC_UNARY_FEATURES:
    desired_type = "NC"
    name = "{}_{}".format(desired_type, f.func_name)
    UNARY_FEATURES.append((name, unary_feature_wrapper, f))
