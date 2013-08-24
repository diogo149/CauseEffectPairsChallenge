from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from functools import partial
from multiprocessing import Pool

from utils import binarize, print_current_time, quick_cache
from convert import NumericalToCategorical, CategoricalToNumerical
from decorators import timer
from unary_features import UNARY_FEATURES
from binary_features import BINARY_FEATURES
from regression_machines import REGRESSION_FEATURES
from classification_machines import CLASSIFICATION_FEATURES


def feature_map(func_name, func, list_of_features):
    new_features = []
    len0 = len(list_of_features[0])
    for features in list_of_features:
        assert len(features) == len0
    for f in zip(*list_of_features):
        names, values = zip(*f)
        feature_name = names[0]
        for n in names:
            assert n == feature_name
        new_name = "{}_{}".format(feature_name, func_name)
        new_features.append((new_name, func(values)))
    return new_features


def feature_difference(f1, f2):
    def diff(values):
        assert len(values) == 2
        return values[0] - values[1]

    return feature_map("difference", diff, [f1, f2])


def feature_sum(list_of_features):
    return feature_map("sum", sum, list_of_features)


def feature_avg(list_of_features):
    return feature_map("average", np.mean, list_of_features)


def combine_features(list_of_features):
    names = []
    values = []
    for features in list_of_features:
        tmp_names, tmp_values = zip(*features)
        names += tmp_names
        values += tmp_values
    return names, values


def preprend_name(pre, features):
    return [("{}_{}".format(pre, name), val) for name, val in features]


def convert_to_categorical(data, data_type):
    assert isinstance(data, np.ndarray)
    NUM_CATEGORIES = 10
    if data_type in ["Binary", "Categorical"]:
        return data
    elif data_type == "Numerical":
        rows = data.shape[0]
        new_data = np.zeros(rows)
        percentile = step = 100.0 / NUM_CATEGORIES
        while percentile < 100.0:
            new_data += data > np.percentile(data, percentile)
            percentile += step
        return new_data
    else:
        raise Exception


def convert_to_numerical(data, data_type):
    assert isinstance(data, np.ndarray)
    if data_type == "Binary":
        return [data]
    elif data_type == "Numerical":
        ss = StandardScaler()
        return [ss.fit_transform(data)]
    elif data_type == "Categorical":
        binarized = binarize(data)
        assert binarized.shape[0] == data.shape[0]
        return list(binarized.T)
    else:
        raise Exception


def preprocess(store):
    FEATURES = BINARY_FEATURES + UNARY_FEATURES + REGRESSION_FEATURES + CLASSIFICATION_FEATURES

    pool = Pool()
    V2_cache = quick_cache("create_V2_cache" + str(store.raw.shape), create_V2_cache, store.raw, pool)
    # V2_cache = create_V2_cache(store.raw, pool)

    features = []
    for feature in FEATURES:
        name, func, func_args = feature[0], feature[1], feature[2:]
        print(name, end=' ')
        print_current_time()
        tmp_feature = store.cache(name, feature_creation_V1, pool, store, func, func_args, name)
        features.append(tmp_feature)
        name2 = "V2_" + name
        tmp_feature2 = store.cache(name2, feature_creation_V2, pool, V2_cache, func, func_args, name)
        tmp_feature2.rename(columns=lambda x: "V2_" + x)
        features.append(tmp_feature2)
    pool.close()
    return pd.concat(features, axis=1)


@timer
def feature_creation_V1(pool, store, func, func_args, name):
    desired_type = name[:2]
    assert desired_type in ["NN", "NC", "CN", "CC"]

    new_func = partial(feature_creation_row_helper, func, func_args, desired_type)

    mapped = pool.map(new_func, store.raw.as_matrix())

    names = None
    transformed = []
    for row_names, transformed_row in mapped:
        if names is None:
            names = row_names
        assert names == row_names
        transformed.append(transformed_row)
    new_names = ["{}_{}".format(name, n) for n in names]
    result = pd.DataFrame(transformed, columns=new_names).fillna(0)
    result[np.isinf(result)] = 0
    return result


def feature_creation_row_helper(func, func_args, desired_type, row):
    if len(func_args) > 0:
        func = func(*func_args)
    return feature_creation_row(func, desired_type, *row)


def feature_creation_row(func, desired_type, x, y, type_x, type_y):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert type_x in ["Numerical", "Categorical", "Binary"]
    assert type_y in ["Numerical", "Categorical", "Binary"]

    left = asymmetric_feature_creation(func, desired_type, x, y, type_x, type_y)
    right = asymmetric_feature_creation(func, desired_type, y, x, type_y, type_x)
    relative = feature_difference(left, right)
    new_left = preprend_name("A->B", left)
    new_right = preprend_name("B->A", right)
    features = (new_left, new_right, relative)
    return combine_features(features)


def asymmetric_feature_creation(func, desired_type, x, y, type_x, type_y):

    cat_x, cat_y = convert_to_categorical(x, type_x), convert_to_categorical(y, type_y)
    num_xs, num_ys = convert_to_numerical(x, type_x), convert_to_numerical(y, type_y)

    if desired_type == "NN":
        nn_tmp = [func(num_x, num_y) for num_x in num_xs for num_y in num_ys]
        features = feature_sum(nn_tmp) + feature_avg(nn_tmp)
    elif desired_type == "CN":
        cn_tmp = [func(cat_x, num_y) for num_y in num_ys]
        features = feature_sum(cn_tmp) + feature_avg(cn_tmp)
    elif desired_type == "NC":
        nc_tmp = [func(num_x, cat_y) for num_x in num_xs]
        features = feature_sum(nc_tmp) + feature_avg(nc_tmp)
    elif desired_type == "CC":
        features = func(cat_x, cat_y)
    else:
        raise Exception("Incorrect desired type: {}".format(desired_type))
    return features


def create_V2_cache_transform(row):
    a, b, a_type, b_type = row
    assert a_type in ["Numerical", "Categorical", "Binary"]
    assert b_type in ["Numerical", "Categorical", "Binary"]
    num_x, cat_x, num_y, cat_y = a, a, b, b
    if a_type == "Numerical":
        cat_x = NumericalToCategorical(verify=False).fit_transform(num_x)
    if a_type == "Categorical":
        num_x = CategoricalToNumerical(verify=False).fit_transform(cat_x)
    if b_type == "Numerical":
        cat_y = NumericalToCategorical(verify=False).fit_transform(num_y)
    if b_type == "Categorical":
        num_y = CategoricalToNumerical(verify=False).fit_transform(cat_y)
    return (num_x, cat_x, num_y, cat_y)


@timer
def create_V2_cache(df, pool):

    assert isinstance(df, pd.DataFrame)
    for col in ['A', 'B', 'A type', 'B type']:
        assert col in df

    V2_cache = pool.map(create_V2_cache_transform, df.as_matrix())
    return tuple(V2_cache)


@timer
def feature_creation_V2(pool, V2_cache, func, func_args, name):
    desired_type = name[:2]
    assert desired_type in ["NN", "NC", "CN", "CC"]

    new_func = partial(feature_creation_row_helper_V2, func, func_args, desired_type)

    mapped = pool.map(new_func, V2_cache)
    # mapped = map(new_func, V2_cache)

    names = None
    transformed = []
    for row_names, transformed_row in mapped:
        if names is None:
            names = row_names
        assert names == row_names
        transformed.append(transformed_row)
    new_names = ["{}_{}".format(name, n) for n in names]
    result = pd.DataFrame(transformed, columns=new_names).fillna(0)
    result[np.isinf(result)] = 0
    return result


def feature_creation_row_helper_V2(func, func_args, desired_type, row):
    if len(func_args) > 0:
        func = func(*func_args)
    row = map(lambda x: x.astype(np.float), row)
    return feature_creation_row_V2(func, desired_type, *row)


def feature_creation_row_V2(func, desired_type, num_x, cat_x, num_y, cat_y):
    assert isinstance(num_x, np.ndarray)
    assert isinstance(cat_x, np.ndarray)
    assert isinstance(num_y, np.ndarray)
    assert isinstance(cat_y, np.ndarray)

    left = asymmetric_feature_creation_V2(func, desired_type, num_x, cat_x, num_y, cat_y)
    right = asymmetric_feature_creation_V2(func, desired_type, num_y, cat_y, num_x, cat_x)
    relative = feature_difference(left, right)
    new_left = preprend_name("A->B", left)
    new_right = preprend_name("B->A", right)
    features = (new_left, new_right, relative)
    return combine_features(features)


def asymmetric_feature_creation_V2(func, desired_type, num_x, cat_x, num_y, cat_y):
    if desired_type == "NN":
        features = func(num_x, num_y)
    elif desired_type == "CN":
        features = func(cat_x, num_y)
    elif desired_type == "NC":
        features = func(num_x, cat_y)
    elif desired_type == "CC":
        features = func(cat_x, cat_y)
    else:
        raise Exception("Incorrect desired type: {}".format(desired_type))
    return features


def metafeature_creation(df):

    def or_(t1, t2):
        return ((t1 + t2) > 0) + 0.0

    def and_(t1, t2):
        return ((t1 + t2) == 2) + 0.0
    types = ["Binary", "Numerical", "Categorical"]
    assert isinstance(df, pd.DataFrame)
    a_type = np.array(df['A type'])
    b_type = np.array(df['B type'])
    metafeatures = []
    columns = []

    for t in types:
        tmp = (a_type == t) + 0.0
        columns.append("aIs" + t)
        metafeatures.append(tmp)

    for t in types:
        tmp = (a_type != t) + 0.0
        columns.append("aIsNot" + t)
        metafeatures.append(tmp)

    for t in types:
        tmp = (b_type == t) + 0.0
        columns.append("bIs" + t)
        metafeatures.append(tmp)

    for t in types:
        tmp = (b_type != t) + 0.0
        columns.append("bIsNot" + t)
        metafeatures.append(tmp)

    for t1 in types:
        for t2 in types:
            tmp = and_(a_type == t1, b_type == t2)
            columns.append("abAre" + t1 + t2)
            metafeatures.append(tmp)
            if t1 <= t2:
                tmp = or_(and_(a_type == t1, b_type == t2), and_(a_type == t2, b_type == t1))
                columns.append("abAreAmong" + t1 + t2)
                metafeatures.append(tmp)

    six_options = or_(a_type == "Binary", b_type == "Binary") + 2 * or_(a_type == "Categorical", b_type == "Categorical") + 3 * and_(a_type == "Binary", b_type == "Binary") + 3 * and_(a_type == "Categorical", b_type == "Categorical")

    columns.append("allTypes")
    metafeatures.append(six_options)

    return metafeatures, columns


def add_metafeatures(df, df_feat):
    metafeatures, columns = metafeature_creation(df)
    assert len(metafeatures) == len(columns)
    for mf, col in zip(metafeatures, columns):
        df_feat["metafeature_" + col] = mf
