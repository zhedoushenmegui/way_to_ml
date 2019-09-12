import os
import sys
import getopt
import math
import pandas as pd
import xgboost
from util.data_process import DataProcess
from util import output_process as dop
import matplotlib.pyplot as plt
from xgboost import plot_importance


class XGBModel:
    """
    y_predict, err, abserr, ape  都不要使用
    所有特征都不可以下划线开头
    """

    def __init__(self, feature_list, params):
        self.feature_list = feature_list
        self.params = params
        self.model = None
        self.info = {}
        self.label_encode_map = {}
        self.f_type_map = {}

    def train(self, x_train, x_test, label_name, weight=None, n_round=1000, early_stopping_rounds=None):
        """

        :param label_name:
        :param early_stopping_rounds:
        :param n_round:
        :param x_test:
        :type x_test: pd.DataFrame
        :param weight:
        :type x_train: pd.DataFrame
        """
        # 特征筛选
        ori_train, ori_test = x_train.copy(), x_test.copy()
        x_train, x_test = x_train[self.feature_list].copy(), x_test[self.feature_list].copy()
        # 编码
        if not self.label_encode_map:
            self.label_encode_map, self.f_type_map = DataProcess.gencode(pd.concat([x_train, x_test]),
                                                                         self.feature_list)
        x_train = DataProcess.encode_process(x_train, self.feature_list, self.label_encode_map)
        x_test = DataProcess.encode_process(x_test, self.feature_list, self.label_encode_map)
        ###
        d_train = xgboost.DMatrix(x_train, label=ori_train[label_name], weight=weight)
        d_test = xgboost.DMatrix(x_test, label=ori_test[label_name])
        ####
        evals = [(d_train, 'train'), (d_test, 'test')]
        self.model = xgboost.train(self.params, d_train, n_round, evals,
                                   early_stopping_rounds=early_stopping_rounds if early_stopping_rounds else n_round)
        ori_train['y_predict'], ori_test['y_predict'] = self.model.predict(d_train), self.model.predict(d_test)
        # 添加
        self.info = {
            'gain_score': XGBModel.gen_gain_score(self.model),
            'weight_score': XGBModel.gen_weight_score(self.model),
            'cover_score': XGBModel.gen_cover_score(self.model),
            'fmap': XGBModel.gen_feat_map(self.f_type_map, self.feature_list),
            'ori_train': ori_train,
            'ori_test': ori_test,
            ###
            'y_train': list(ori_train[label_name]),
            'y_test': list(ori_test[label_name]),
        }

    @staticmethod
    def gen_feat_map(f_type_map, feature_list):
        return [(i, feature, f_type_map.get(feature))
                for i, feature in enumerate(feature_list)]

    @staticmethod
    def gen_cover_score(model):
        return sorted(model.get_score(importance_type='cover').items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def gen_weight_score(model):
        return sorted(model.get_score(importance_type='weight').items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def gen_gain_score(model):
        return sorted(model.get_score(importance_type='gain').items(), key=lambda x: x[1], reverse=True)

    # 预测
    def predict(self, X, y=None):
        en_x = DataProcess.encode_process(X[self.feature_list], self.feature_list, self.label_encode_map)
        dx = xgboost.DMatrix(en_x, label=y)
        ###
        y_predict = self.model.predict(dx)
        ###
        return list(y_predict)

    def dump_info(self, folder_path='./', name=''):
        name = name.strip().replace(" ", "_")
        s_name = ("_" if name else "") + name
        dop.dump_label_encode_map(
            self.label_encode_map,
            f'{folder_path}/label_encode{s_name}.json'
        )
        dop.dump_XGB_score(
            self.info['gain_score'],
            f'{folder_path}/gain{s_name}.txt',
        )
        dop.dump_XGB_score(
            self.info['weight_score'],
            f'{folder_path}/weight{s_name}.txt',
        )
        dop.dump_XGB_score(
            self.info['cover_score'],
            f'{folder_path}/covers{s_name}.txt',
        )
        # 二进制模型文件 bin 的文件名不能有空格 替换为 _
        dop.save_XGBModel(
            self.model,
            f'{folder_path}/bmodel{s_name}.bin'
        )
        dop.dump_XGBModel(
            self.model,
            f'{folder_path}/model{s_name}.txt'
        )
        dop.dump_feature_map(
            self.info['fmap'],
            f'{folder_path}/fmap{s_name}.txt'
        )
        fig, ax = plt.subplots(figsize=(10, 15))
        plot_importance(self.model, height=0.5, max_num_features=64, ax=ax)
        plt.savefig(f'{folder_path}/feature{s_name}.png')
