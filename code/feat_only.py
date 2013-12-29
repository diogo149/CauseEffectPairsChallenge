from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.metrics import auc_score
from sklearn.ensemble import GradientBoostingRegressor
from csv import writer as csv_writer
from collections import defaultdict
from copy import deepcopy

from feature_cache import FeatureCache
from random_functions import preprocess, add_metafeatures


import SETTINGS
from utils import quick_load


def parse_dataframe(filename):
    df = pd.read_csv(filename)
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df


def combine_data_and_types(df_data, df_types):
    assert isinstance(df_data, pd.DataFrame)
    assert isinstance(df_types, pd.DataFrame)
    assert df_data.shape[0] == df_types.shape[0]
    return pd.DataFrame([df_data['A'], df_data['B'], df_types['A type'], df_types['B type']]).T


def get_df(filename):
    data = parse_dataframe(filename)
    types = pd.read_csv(filename.replace("pairs", "publicinfo"))
    return combine_data_and_types(data, types)


def double_data(df):
    tmp = 0 * deepcopy(df)
    for col_name in df.columns:
        if "A->B" in col_name:
            tmp[col_name.replace("A->B", "B->A")] = df[col_name]
        elif "B->A" in col_name:
            tmp[col_name.replace("B->A", "A->B")] = df[col_name]
        elif "difference" in col_name:
            tmp[col_name] = -df[col_name]
        else:
            raise Exception
    big_df = pd.concat((df, tmp))
    big_df.index = range(big_df.shape[0])
    return big_df


def double_original_data(df):
    tmp = 0 * deepcopy(df)
    for col_name in df.columns:
        if "A" in col_name:
            tmp[col_name.replace("A", "B")] = df[col_name]
        elif "B" in col_name:
            tmp[col_name.replace("B", "A")] = df[col_name]
        else:
            raise Exception
    big_df = pd.concat((df, tmp))
    big_df.index = range(big_df.shape[0])
    return big_df

if __name__ == "__main__":
    np.random.seed(1)
    train_file = SETTINGS.FC_TRAIN.TRAIN_FILE
    train_file = "../data/CEfinal_train_text/CEfinal_train_pairs.csv"
    train = get_df(train_file)

    train_store = FeatureCache(train)
    train_feat = preprocess(train_store)

    # valid_file = SETTINGS.FC_TRAIN.VALID_FILE
    # valid_file = "../data/CEfinal_test_text/CEfinal_test_pairs.csv"
    # valid = get_df(valid_file)

    # valid_store = FeatureCache(valid)
    # valid_feat = preprocess(valid_store)

    # target_df = pd.read_csv(train_file.replace("pairs", "target"))
    # target_col = [col for col in target_df if "Target" in col]
    # assert len(target_col) == 1
    # target = target_df[target_col]

    double = True
    if double:
        train_feat = double_data(train_feat)
        # valid_feat = double_data(valid_feat)
        # target = np.vstack((target, -target))
        train = double_original_data(train)
        # valid = double_original_data(valid)

    add_metafeatures(train, train_feat)
    # add_metafeatures(valid, valid_feat)
