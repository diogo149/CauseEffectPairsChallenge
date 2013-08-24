import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import RandomizedPCA

from utils import is_categorical
from gap_statistic import FittedMiniBatchKMeans


class NumericalToCategorical(object):

    def __init__(self, clustering=None, min_clusters=2, verify=True):
        """Takes in a clustering classifier in order to convert numerical features into categorical.
        """
        if clustering is None:
            clustering = FittedMiniBatchKMeans(min_clusters)
        self.clustering = clustering
        self.verify = verify

    def fit(self, X, y=None):
        self._verify(X, self.verify)
        reshaped = X.reshape(-1, 1)
        self.clustering.fit(reshaped)

    def transform(self, X):
        self._verify(X, False)
        reshaped = X.reshape(-1, 1)
        result = self.clustering.predict(reshaped)
        assert result.shape == X.shape
        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _verify(self, X, verify):
        if verify:
            assert not is_categorical(X)
        else:
            assert isinstance(X, np.ndarray)
            assert len(X.shape) == 1


class CategoricalToNumerical(object):

    def __init__(self, dimensionality_reducer=None, verify=True):
        pass
        """Takes in a dimensionality reducer in order to convert categorical features into numerical.
        """
        if dimensionality_reducer is None:
            dimensionality_reducer = RandomizedPCA(1)
        self.dimensionality_reducer = dimensionality_reducer
        self.verify = verify
        self.binarizer = LabelBinarizer()

    def fit(self, X, y=None):
        self._verify(X, self.verify)
        binarized = self.binarizer.fit_transform(X)
        self.dimensionality_reducer.fit(binarized)

    def transform(self, X):
        self._verify(X, False)
        binarized = self.binarizer.transform(X)
        result = self.dimensionality_reducer.transform(binarized).flatten()
        assert X.shape == result.shape
        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def _verify(self, X, verify):
        if verify:
            assert is_categorical(X)
        else:
            assert isinstance(X, np.ndarray)
            assert len(X.shape) == 1
