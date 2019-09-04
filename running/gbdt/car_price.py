from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import pandas as pd
import numpy as np
import os
import sys
from util.data_process import DataProcess


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
    ####
    est = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.3, max_depth=6, min_samples_leaf=20,
                                        max_leaf_nodes=40)
    est.fit(en_train, d_train.price)
    pred = est.predict(en_test)
    evaluate(d_test, pred)
    ### R2
    print(est.score(en_test, d_test.price))
