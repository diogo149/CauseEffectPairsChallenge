CauseEffectPairsChallenge
=========================

Name: Diogo Moitinho de Almeida
Kaggle ID: Dee5
email: diogo149@gmail.com
Team: ProtoML

Software Used:
    arch linux (for feature creation)
        python 2.7.5
        numpy
        scipy
        scikit-learn
        pandas
        ipython
    ubuntu 12.04 (for hyperparameter optimization)
        python 2.7.3
        numpy
        scipy
        scikit-learn
        pandas
        ipython

Package Versions:
    -numpy 1.7.1
    -scipy 0.12.0
    -pandas 0.11.0
    -scikit-learn 0.13.1
    -ipython 0.13.2

Hardware needed:
    -feature creation will probably take +5GB
    -running on the entire dataset took several days on an 8 core machine
    -about 4GB of RAM per core was needed

To run with training:
    -open an ipython terminal
    -run:
    >>> %time %run fc_train.py

To run with testing only:
    -open an ipython terminal
    -run:
    >>> %time %run test_only.py


Notes:
    -The relevant settings can be changed in SETTINGS.py

For my 3 submissions, I use settings:

    Getting leaderboard score: 0.81367
        FC_TRAIN.USE_ALL_FEAT = False
        FC_TRAIN.USE_NON_GA_FEAT = False
        FC_TRAIN.CLF = GradientBoostingRegressor(loss='huber', n_estimators=5000, random_state=1, min_samples_split=2, min_samples_leaf=1, subsample=1.0, max_features=686, alpha=0.995355212043, max_depth=10, learning_rate=np.exp(-4.09679792914))

    Getting leaderboard score: 0.81279
        FC_TRAIN.USE_ALL_FEAT = True
        FC_TRAIN.USE_NON_GA_FEAT = False
        FC_TRAIN.CLF = GradientBoostingRegressor(loss='huber', n_estimators=5000, random_state=1, min_samples_split=2, min_samples_leaf=1, subsample=1.0, max_features=500, alpha=0.95, max_depth=10, learning_rate=np.exp(-3.28469694591))

    Getting leaderboard score: 0.81238
        FC_TRAIN.USE_ALL_FEAT = True
        FC_TRAIN.USE_NON_GA_FEAT = False
        FC_TRAIN.CLF = GradientBoostingRegressor(loss='huber', n_estimators=5000, random_state=1, min_samples_split=2, min_samples_leaf=1, subsample=1.0, max_features=686, alpha=0.99517924408, max_depth=10, learning_rate=np.exp(-4.10031144415))
