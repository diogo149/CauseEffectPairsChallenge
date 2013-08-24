import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from utils import binarize
from classification_metrics import CLASSIFICATION_METRICS

CLASSIFICATION_MACHINES = (
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    GradientBoostingClassifier(subsample=0.5, n_estimators=10, random_state=0),
    KNeighborsClassifier(),
    GaussianNB(),
)


def classification_features(clf, binarize_x=False):
    def inner_func(x, y):
        if binarize_x:
            x = binarize(x)
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        assert len(x.shape) == 2
        unique_y = np.unique(y)
        has_one_class = (len(unique_y) < 2)
        if has_one_class:
            pred = y
            scores = [0]
        else:
            clf.fit(x, y)
            pred = clf.predict(x)
        clf_feat = []
        for metric in CLASSIFICATION_METRICS:
            metric_name = metric.func_name
            if not has_one_class:
                scores = []
                for cls in unique_y:
                    scores.append(metric((y == cls) + 0.0, (pred == cls) + 0.0))
            clf_feat.append((metric_name + "_avg", np.mean(scores)))
            clf_feat.append((metric_name + "_sum", sum(scores)))
            clf_feat.append((metric_name + "_max", max(scores)))
        return clf_feat
    return inner_func

CLASSIFICATION_FEATURES = []
for clf in CLASSIFICATION_MACHINES:
    clf_name = str(clf.__class__)
    CLASSIFICATION_FEATURES.append(("NC_" + clf_name, classification_features, clf))
    CLASSIFICATION_FEATURES.append(("CC_" + clf_name, classification_features, clf, True))
