
class Setting(object):
    pass

MISC = Setting()
MISC.WRITE = False  # set to true to enable caching

IS_CATEGORICAL = Setting()
IS_CATEGORICAL.THRESHOLD = 0.1

GAP_STATISTIC = Setting()
GAP_STATISTIC.RANDOMIZED_PCA_THRESHOLD = 10
GAP_STATISTIC.NUM_CLUSTERS_WITHOUT_IMPROVEMENT = 5
GAP_STATISTIC.MAXIMUM_DECLINE = 0.5

QUICK_CACHE = Setting()
QUICK_CACHE.DIRECTORY = "quick_cache"

FC_TRAIN = Setting()
FC_TRAIN.TRAIN_FILE = "../data/combination/pairs.csv"
FC_TRAIN.VALID_FILE = "../data/CEfinal/CEfinal_valid_pairs.csv"
FC_TRAIN.LOCAL_VALID_FILE = "../data/CEfinal/CEfinal_train_pairs.csv"
FC_TRAIN.SUBMISSION = "submission.csv"

# this has priority
# if both are false, then GA feat is used
FC_TRAIN.USE_ALL_FEAT = True
FC_TRAIN.USE_NON_GA_FEAT = False

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
FC_TRAIN.CLF = GradientBoostingRegressor(loss='huber', n_estimators=5000, random_state=1, min_samples_split=2, min_samples_leaf=1, subsample=1.0, max_features=686, alpha=0.995355212043, max_depth=10, learning_rate=np.exp(-4.09679792914))

# dummy classifier
from sklearn.dummy import DummyRegressor
FC_TRAIN.CLF = DummyRegressor()

TEST_ONLY = Setting()
TEST_ONLY.CLF_DIR = "../clf"

MODEL_NUMER = 2  # if you want to use model 1, download from S3: "https://s3-us-west-2.amazonaws.com/causeeffectpairs/code/clfs/5kfeat_ga_subset3_(48060, 4257).pickle"

if MODEL_NUMER == 1:
    FC_TRAIN.USE_ALL_FEAT = False
    TEST_ONLY.CLF_NAME = "5kfeat_ga_subset3_(48060, 4257)"
elif MODEL_NUMER == 2:
    TEST_ONLY.CLF_NAME = "5kfeat_(48060, 8563)"
elif MODEL_NUMER == 3:
    FC_TRAIN.USE_ALL_FEAT = False
    TEST_ONLY.CLF_NAME = "5kfeat_ga_subset2_(48060, 4257)"
else:
    raise Exception("Incorrect model number")
