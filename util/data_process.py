import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
"""
数据处理相关的方法
"""

class DataProcess:
    @staticmethod
    def gencode(df, feature_list):
        """
        生成编码结果
        :param df:
        :param feature_list:
        :return:
        """
        le = LabelEncoder()
        label_encode_map = {}
        f_type_map = {}
        for f in feature_list:
            try:
                temp = df[f]
                if temp.dtype == 'object':
                    # 缺省值用-1 填充
                    df.loc[:, f] = le.fit(df[f].fillna('-1').astype(str))
                    label_encode_map[f] = {y: x for x, y in enumerate(le.classes_)}
                else:
                    df.loc[:, f] = pd.to_numeric(df[f].fillna('-1')).astype(float)
            except Exception:
                print(f'{f}')
                df.loc[:, f] = le.fit(df[f].fillna('-1').astype(str))
                label_encode_map[f] = {y: x for x, y in enumerate(le.classes_)}
            f_type_map[f] = 'q'
        return label_encode_map, f_type_map

    @staticmethod
    def encode_process(df, feature_list, label_encode_map):
        df_copy = df.copy()
        for feature, mapping in label_encode_map.items():
            if feature in feature_list:
                df_copy[feature] = df_copy[feature].map(mapping).fillna(-1)
        return df_copy