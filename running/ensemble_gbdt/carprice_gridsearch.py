from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from util.data_process import DataProcess
import os
import sys
from running.gbdt.car_price import evaluate


if __name__ == '__main__':
    project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('.')), '../'))
    sys.path.append(project_path)
    ###
    with open('car_price_feat.txt') as f:
        feat_list = list(filter(lambda x: x[0] != '#', f.read().split('\n')))
    ###
    data_train = pd.read_csv(f'{project_path}/data/car_price_train.201908.csv')
    data_test = pd.read_csv(f'{project_path}/data/car_price_test.201908.csv')
    ###
    series_name = '宝马5系'
    d_train = data_train[data_train.model_series == series_name]
    d_test = data_test[data_test.model_series == series_name]
    ###
    label_encode_map, f_map = DataProcess.gencode(pd.concat([data_train, data_test]), feat_list)
    en_train, en_test = DataProcess.encode_process(d_train[feat_list], feat_list,
                                                   label_encode_map), DataProcess.encode_process(d_test[feat_list],
                                                                                                 feat_list,
                                                                                                 label_encode_map)
    ###
    param_grid = {
        "learning_rate": [0.1, 0.2, 0.4, 0.05],
        "max_depth": [3, 4, 5],
        "subsample": [0.7, 0.8, 0.9]
    }
    gbt = GradientBoostingRegressor(min_samples_leaf=5, min_samples_split=11, max_depth=3, max_leaf_nodes=30,
                                    subsample=0.8, n_estimators=250)
    gscv = GridSearchCV(gbt, param_grid, cv=5)
    gscv.fit(en_train, d_train.price)





