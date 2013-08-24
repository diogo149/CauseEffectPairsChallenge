from __future__ import print_function
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import auc_score
from collections import defaultdict

from storage import quick_load


def target_score(y_true, predictions):
    return (auc_score(y_true == 1, predictions) + auc_score(y_true == -1, -predictions)) / 2


def output_summary(y_true, predictions, publicinfo):
    print("total error:\t{}".format(target_score(y_true, predictions)))
    indices = defaultdict(list)
    for idx in range(publicinfo.shape[0]):
        key = tuple(sorted([publicinfo['A type'][idx], publicinfo['B type'][idx]]))
        indices[key].append(idx)
    for key, idx in indices.items():
        print("{}:\t{}".format(key, target_score(y_true[idx], predictions[idx])))

if __name__ == "__main__":
    train_feat = quick_load("model_selection", "train_feat")
    valid_feat = quick_load("model_selection", "valid_feat")
    target = quick_load("model_selection", "target")
    y_true = quick_load("model_selection", "y_true")
    valid_types = quick_load("model_selection", "valid_types")

    if 0:
        bad_indices = list(range(7831)) + list(range(19980, 19980 + 7831))
        train_feat = np.delete(train_feat.as_matrix(), bad_indices, axis=0)
        target = np.delete(target, bad_indices, axis=0)
        y = target.flatten()[:12149]

        np.random.seed(42)
        clf = GradientBoostingRegressor(loss='huber', n_estimators=1000, random_state=1, min_samples_split=2, min_samples_leaf=1, subsample=0.5, max_features=92)

        clf.fit(train_feat, target)
        predictions = clf.predict(valid_feat)
        pred_len = predictions.shape[0] / 2
        predictions = predictions[:pred_len] - predictions[pred_len:]
        output_summary(y_true, predictions, valid_types)

        from utils import cv_fit_predict
        predictions2 = cv_fit_predict(clf, train_feat, target.flatten(), n_folds=3, n_jobs=3)
        pred_len = predictions2.shape[0] / 2
        predictions2 = predictions2[:pred_len] - predictions2[pred_len:]
        output_summary(y[-6151:-162], predictions2[-6151:-162], valid_types)
    else:
        for i in [100 * i for i in range(1, 11)]:
            print(i)
            np.random.seed(42)
            clf = GradientBoostingRegressor(loss='huber', n_estimators=i, random_state=1, min_samples_split=2, min_samples_leaf=1, subsample=0.5, max_features=92)

            clf.fit(train_feat, target)
            predictions = clf.predict(valid_feat)
            pred_len = predictions.shape[0] / 2
            predictions = predictions[:pred_len] - predictions[pred_len:]
            output_summary(y_true, predictions, valid_types)
