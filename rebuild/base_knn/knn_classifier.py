import numpy as np
import pandas as pd
from rebuild.classifier import Classifier, label_key
import random
import os
import sys
from util.process_bar import ProcessBar
import time

project_path = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class KnnClassifier(Classifier):
    def __init__(self, k, row_samples=1.0):
        super().__init__()
        self.test_pred_score = []
        self.test_preds = []
        self.k = k
        self.train_num = 0
        # 行采样
        self.row_samples = row_samples
        self.feats = []
        self.train_X_norm = None
        self.test_X_norm = None

    def fit_process(self):
        self.feats = list(self.train_X.columns)
        # 采样
        if self.row_samples < 1.0:
            self.train_X[label_key] = self.train_y
            self.train_X = self.train_X.sample(frac=self.row_samples)
            self.train_y = list(self.train_X[label_key])
        self.train_num = len(self.train_X)
        #
        self.train_X_norm = pd.DataFrame()
        for f in self.feats:
            self.train_X_norm[f'{f}'] = KnnClassifier.auto_norm(self.train_X[f])
        self.train_X_norm[label_key] = self.train_y
        return

    @staticmethod
    def auto_norm(arr):
        """
        归一化
        :param arr:
        :return:
        """
        _max = max(arr)
        _min = min(arr)
        _size = _max - _min
        return list(map(lambda x: (x-_min)/_size, arr))

    def test_process(self):
        #
        self.test_X_norm = pd.DataFrame()
        for f in self.feats:
            self.test_X_norm[f'{f}'] = KnnClassifier.auto_norm(self.test_X[f])
        self.test_X_norm[label_key] = self.test_y
        # 计算距离
        ts_dict = self.test_X_norm.to_dict()
        #
        preds = []
        pred_score = []
        #
        pb = ProcessBar(len(self.test_X))
        run_num = 0
        #
        for i in ts_dict[self.feats[0]]:
            ndf = pd.DataFrame()
            for f in self.feats:
                ndf[f] = (self.train_X_norm[f] - ts_dict[f][i]) ** 2
            ndf['cal_score_cal_score_val'] = sum([ndf[f] for f in self.feats])
            ndf[label_key] = self.train_y
            ###
            ndf.sort_values(by=['cal_score_cal_score_val'], inplace=True)
            mdf = ndf.head(self.k)
            ###
            rdict = mdf.groupby(label_key).count().to_dict()
            _size = len(mdf)
            _max = 0
            _bestlabel = None
            for label in rdict['cal_score_cal_score_val']:
                _n = rdict['cal_score_cal_score_val'][label]
                if _bestlabel is None or _n > _max:
                    _max = _n
                    _bestlabel = label
            ###
            preds.append(_bestlabel)
            pred_score.append(_max/_size)
            ###
            run_num += 1
            pb.update(run_num)
        pb.end(f'用时: {round(time.time()-pb.start_ts, 3)}')
        self.test_preds = preds
        self.test_pred_score = pred_score


if __name__ == '__main__':
    # 加载数据
    date_df = pd.read_csv(project_path + '/data/datingTestSet2.txt')
    # 拆分训练集和测试集
    size = len(date_df)
    tn = date_df[date_df.index < .6 * size]
    ts = date_df[(date_df.index >= .6 * size) & (date_df.index < .9 * size)]
    # 输入特征和训练集, knn 不存在训练过程, fit 只存入标注集
    feats = ['a', 'b', 'c']
    # k 是 40
    knn = KnnClassifier(40)
    knn.fit(tn[feats], list(tn.label))
    #
    knn.test(ts[feats], list(ts.label))
    #
    result = knn.cal_accuracy(knn.test_y, knn.test_preds)
    print(f'测试集数量: {result[0]}')
    print(f'正确分类数量: {result[1]}')
    print(f'准确率: { round(result[2]*100, 2)} %')
    exit()
    # 以下是遍历20 - 100 的k, 来分析k 的最优值
    for k in range(20, 100, 2):
        knn = KnnClassifier(k, .8)
        knn.fit(tn[feats], list(tn.label))
        #
        knn.test(ts[feats], list(ts.label))
        #
        result = knn.cal_accuracy(knn.test_y, knn.test_preds)
        print(f'{k}: { round(result[2]*100, 2)} %')


