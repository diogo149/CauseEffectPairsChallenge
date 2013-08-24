"""
to do:
    -predict_proba
"""
import numpy as np
import pandas as pd

from copy import deepcopy

from parallel import parmap


def get_fitted_clf(clf, X, y, item):
    clf = deepcopy(clf)
    val, indices = item
    clf.fit(X[indices], y[indices])
    return (val, clf)


def predict_with_clf(X, item):
    val, clf = item
    return (val, clf.predict(X))


class BinningMachine(object):

    def __init__(self, clf, metafeature, n_jobs=1):
        self.clf = clf
        self.metafeature = metafeature
        self.n_jobs = n_jobs

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame)
        assert self.metafeature in X

        indices = self.indices(X)
        mapped = parmap(get_fitted_clf, indices, (self.clf, X, y), n_jobs=self.n_jobs)
        self.clfs = dict(mapped)

    def predict(self, X):
        assert isinstance(X, pd.DataFrame)
        assert self.metafeature in X

        predictions = self.predict_multi(X)

        col = X[self.metafeature]
        output = np.zeros(X.shape[0])
        for val in self.clfs:
            output += np.array(col == val) * predictions[val]
        return output

    def predict_multi(self, X):
        mapped = parmap(predict_with_clf, self.clfs.items(), [X], n_jobs=self.n_jobs)
        return pd.DataFrame(dict(mapped))

    def predict_proba(self, X):
        # TODO
        pass

    def indices(self, X):
        return [(val, X[self.metafeature] == val) for val in np.unique(X[self.metafeature])]


class NumpyBinningMachine(object):

    def __init__(self, clf, column_num, n_jobs=1):
        self.binning_machine = BinningMachine(clf, column_num, n_jobs=1)

    def fit(self, X, y):
        self.binning_machine.fit(pd.DataFrame(X), y)

    def predict(self, X):
        return self.binning_machine.predict(pd.DataFrame(X))

    def predict_multi(self, X):
        return self.binning_machine.predict_multi(pd.DataFrame(X))

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    np.random.seed(1)
    X = pd.DataFrame(np.random.randn(4, 5))
    y = np.random.randn(4)
    X["doo"] = (X[0] > 0) + 0.0

    clf = BinningMachine(LinearRegression(), "doo", n_jobs=1)
    clf.fit(X, y)
    print clf.predict(X)
    print y
