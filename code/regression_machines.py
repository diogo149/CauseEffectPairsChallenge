from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


from utils import binarize
from regression_metrics import REGRESSION_METRICS
from unary_features import ALL_UNARY_FEATURES, NN_UNARY_FEATURES, unary_feature_wrapper
from binary_features import ALL_BINARY_FEATURES, NN_BINARY_FEATURES, binary_feature_wrapper


REGRESSION_MACHINES = (
    Ridge(),
    LinearRegression(),
    DecisionTreeRegressor(random_state=0),
    RandomForestRegressor(random_state=0),
    GradientBoostingRegressor(subsample=0.5, n_estimators=10, random_state=0),
    KNeighborsRegressor(),
)


def regression_features(clf, binarize_x=False):
    def inner_func(x, y):
        if binarize_x:
            x = binarize(x)
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        assert len(x.shape) == 2
        clf.fit(x, y)
        pred = clf.predict(x)
        clf_feat = []
        for metric in REGRESSION_METRICS:
            metric_name = metric.func_name
            score = metric(y, pred)
            clf_feat.append((metric_name, score))
        for f in ALL_BINARY_FEATURES + NN_BINARY_FEATURES:
            clf_feat += binary_feature_wrapper(f)(y, pred)
        for f in ALL_UNARY_FEATURES + NN_UNARY_FEATURES:
            clf_feat += unary_feature_wrapper(f)(y - pred, None)

        return clf_feat
    return inner_func

REGRESSION_FEATURES = []
for clf in REGRESSION_MACHINES:
    clf_name = str(clf.__class__)
    REGRESSION_FEATURES.append(("NN_" + clf_name, regression_features, clf))
    REGRESSION_FEATURES.append(("CN_" + clf_name, regression_features, clf, True))
