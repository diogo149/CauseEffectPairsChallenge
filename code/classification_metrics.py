from sklearn.metrics import accuracy_score, auc_score, average_precision_score, f1_score, hinge_loss, matthews_corrcoef, precision_score, recall_score, zero_one_loss
from collections import Counter


def categorical_gini_coefficient(x):
    len_x = len(x)
    counter = Counter(x)
    total = 0.0
    for _, count in counter.items():
        total += len_x - count
    return total / (len_x ** 2)


def categorical_gini_loss(y_true, y_pred):
    # this is kind of random
    return categorical_gini_coefficient(y_true != y_pred)

CLASSIFICATION_METRICS = (
    accuracy_score,
    auc_score,
    average_precision_score,
    f1_score,
    hinge_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    zero_one_loss,
    categorical_gini_loss,
)
