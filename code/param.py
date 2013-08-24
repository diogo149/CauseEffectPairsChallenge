from __future__ import print_function
# THIS IS A TEST
import cPickle as pickle
# import pickle


class Param(object):

    """
    Class that allows you to store values when training and retrieve values when testing as either a function result, or a value.
    """

    class ParamStore(object):

        def __init__(self, filename):
            self.filename = filename

        def __enter__(self):
            assert Param.instance is None
            Param.instance = self
            try:
                with open(self.filename) as infile:
                    self.values = pickle.load(infile)
            except IOError:
                self.values = {}

        def __exit__(self, type, value, traceback):
            Param.instance = None
            with open(self.filename, 'w') as outfile:
                pickle.dump(self.values, outfile)

        def __getitem__(self, key):
            return self.values[key]

        def __setitem__(self, key, value):
            self.values[key] = value

    instance = None
    trainMode = False

    @staticmethod
    def v(identifier, value):
        """
        Stores a value with an identifier.
        """
        return Param.f(identifier, lambda: value)

    @staticmethod
    def f(identifier, func, *args, **kwargs):
        """
        Stores a function result with an identifier.
        """
        if Param.instance is None:
            return func(*args, **kwargs)
        if Param.trainMode:
            Param.instance[identifier] = func(*args, **kwargs)
        return Param.instance[identifier]

    @staticmethod
    def train(filename):
        """
        Sets global Param mode to train.
        """
        Param.trainMode = True
        return Param.ParamStore(filename)

    @staticmethod
    def test(filename):
        """
        Sets global Param mode to test.
        """
        Param.trainMode = False
        return Param.ParamStore(filename)

"""
The following are shortcuts. You can call them with:
    >>> import param
    >>> param.f(identifier, func)
Instead of:
    >>> import param
    >>> param.Param.f(identifier, func)
"""
train = Param.train
test = Param.test
f = Param.f
v = Param.v


class SETTINGS(object):

    """
    Class used to store settings. Access with:
        >>> some_variable = SETTINGS.[name]
            OR
        >>> SETTINGS.[name] = value
    """

    @staticmethod
    def default(**kwargs):
        """
        Allows default settings to be set, if not yet set. Only takes in keyword arguments.
        """
        for name, default_val in kwargs.items():
            try:
                getattr(SETTINGS, name)
            except AttributeError:
                setattr(SETTINGS, name, default_val)
