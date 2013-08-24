from __future__ import print_function
import numpy as np
import pandas as pd
import warnings


from utils import hash_df, quick_load, quick_save, try_mkdir


class FeatureCache(object):

    def __init__(self, df):
        self._rows = df.shape[0]
        self._directory = hash_df(df)
        try_mkdir(self._directory)
        self.raw = df

    def validate(self, df, is_safe=True):
        assert isinstance(df, pd.DataFrame), df.__class__
        assert len(df.shape) == 2, df.shape
        assert df.shape[0] == self._rows, (df.shape, self._rows)
        if is_safe:
            try:
                df.astype(np.float)
            except ValueError:
                assert False, "DataFrame is not numeric in FeatureCache."
            assert not np.any(pd.isnull(df))
            assert not np.any(np.isinf(df))

    def _put(self, is_safe, name, func, *args, **kwargs):
        result = func(*args, **kwargs)
        self.validate(result, is_safe=is_safe)
        result = result.astype(np.float) if is_safe else result
        quick_save(self._directory, name, result)
        return result

    def put_unsafe(self, name, func, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._put(False, name, func, *args, **kwargs)

    def put(self, name, func, *args, **kwargs):
        return self._put(True, name, func, *args, **kwargs)

    def get(self, name):
        return quick_load(self._directory, name)

    def cache(self, name, func, *args, **kwargs):
        try:
            return self.get(name)
        except:
            print("Feature Cache Miss: {}".format(name))
            return self.put(name, func, *args, **kwargs)
