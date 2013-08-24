

class PredictReshaper(object):

    def __init__(self, clf):
        self.clf = clf

    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs).reshape(-1, 1)

    def __getattr__(self, name):
        return getattr(self.clf, name)
