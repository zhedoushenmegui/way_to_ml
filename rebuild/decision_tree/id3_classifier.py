import pandas as pd
import numpy as np
import collections as cl
import math
from rebuild.classifier import Classifier, label_key
import os

project_path = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class DtNode:
    def __init__(self):
        self.feat = ''
        # 特征值: node; 特征值: label
        self.children = {}

    def str_process(self):
        result = []
        for x in self.children:
            if isinstance(self.children[x], DtNode):
                rst = self.children[x].str_process()
                for line in rst:
                    result.append(f'[{self.feat}: {x}] {line}')
            else:
                result.append(f'[{self.feat}: {x}] [label: {self.children[x]}]')
        return result


class Id3Classifier(Classifier):
    def __init__(self, feats, max_depth=10, min_node_samples=10, min_gain_split=0.01, silent=False):
        """

        :param feats:
        :param max_depth:
        :param min_node_samples:
        """
        super().__init__()
        # 最大深度, 最小样本数, 最小分裂增益
        # 用来防止过拟合
        self.max_depth = max_depth
        self.min_node_samples = min_node_samples
        self.min_gain_split = min_gain_split
        #
        self.root = None
        self.feats = feats
        #
        self.silent = silent

    def fit_process(self):
        train_df = self.train_X[self.feats].copy()
        train_df[label_key] = self.train_y
        ##
        self.root = self.dfs_create_tree(train_df)
        a = 1

    def dfs_create_tree(self, df, depth=1):
        """
        深度优先遍历生成决策树
        :param depth:
        :type df: pd.DataFrame
        :param df:
        :return:
        """
        # 树的深度过深 不再分裂
        if depth > self.max_depth:
            return {'label': Id3Classifier.choose_most_num_label(df)}
        # 样本过少 不再分裂,
        if len(df) < self.min_node_samples:
        #
        best_feat, base_ent, ent_gain = Id3Classifier.choose_best_feat_by_info_gain(df)
        #
        uniq_vals = set(df[best_feat])
        # 只有一个特征值, 停止分裂
        if len(uniq_vals) < 2 or ent_gain < self.min_gain_split:
            return {'label': Id3Classifier.choose_most_num_label(df)}
        ###
        dt_node = DtNode()
        dt_node.feat = best_feat
        for val in uniq_vals:
            tmp = self.dfs_create_tree(df[df[best_feat] == val], depth + 1)
            dt_node.children[val] = tmp
        return dt_node

    @staticmethod
    def choose_most_num_label(df, labelkey=label_key):
        counter = cl.Counter()
        for x in list(df[labelkey]):
            counter[x] += 1
        _max = 0
        _bestlabel = None
        for x in counter:
            if _bestlabel is None or counter[x] > _max:
                _bestlabel = x
                _max = counter[x]
        return _bestlabel

    @staticmethod
    def cal_entropy(arr):
        """
        计算数组的香农熵
        :param arr:
        :return:
        """
        counter = cl.Counter()
        n = len(arr)
        ent = 0
        for x in arr:
            counter[x] += 1
        for x in counter:
            p = counter[x] / n
            ent -= p * math.log2(p)
        return ent

    @staticmethod
    def choose_best_feat_by_info_gain(df):
        """
        根据熵增益划分数据
        :param df:
        :return:
        """
        keys = list(df.columns)
        base_ent = Id3Classifier.cal_entropy(df[keys[-1]])
        best_ent, best_feat = base_ent, None
        #
        for x in keys[:-1]:
            new_ent = 0
            uniq_ele = set(df[x])
            for ele in uniq_ele:
                _ent = Id3Classifier.cal_entropy(list(df[df[x] == ele][keys[-1]]))
                new_ent += _ent * len(df[df[x] == ele]) / len(df)
            if best_feat is None or new_ent < best_ent:
                best_ent = new_ent
                best_feat = x
        return best_feat, base_ent, base_ent - best_ent


if __name__ == '__main__':
    df = pd.read_csv(project_path + '/data/lenses.txt')
    ##
    feats = 'age,prescript,astigmatic,tearRate'.split(',')
    id3_dt = Id3Classifier(feats, min_node_samples=2, min_gain_split=0.01)
    id3_dt.fit(df, df.label)
    rst = id3_dt.root.str_process()
    for s in rst:
        print(s)
