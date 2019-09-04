import numpy as np
import pandas as pd
import copy

label_key = 'y_label_y_label_tag'


class Classifier:
    test_X: pd.DataFrame
    train_X: pd.DataFrame

    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.feats = []
        ###
        self.eval_result = {
            'recall': 0.0,
            'precision': 0.0,
            'accuracy': 0.0,
            'f1': 0.0,
            'auc': 0.0,
        }
        self.eval_result_train = copy.deepcopy(self.eval_result)

    def set_feats(self, feats):
        self.feats = feats

    def fit(self, train_X, train_y):
        self.train_X = train_X.copy()
        self.train_y = train_y
        self.fit_process()

    def fit_process(self):
        pass

    def test(self, test_X, test_y):
        self.test_X = test_X.copy()
        self.test_y = test_y
        self.test_process()

    def test_process(self):
        pass

    def predict(self, X):
        pass

    def evaluate(self, labels, preds):
        pass

    @staticmethod
    def cal_accuracy(labels, preds):
        _s = len(labels)
        _t = sum([1 if t[0] == t[1] else 0 for t in zip(labels, preds)])
        return _t, _s, _t/_s