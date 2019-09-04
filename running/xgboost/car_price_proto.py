import pandas as pd
import numpy as np
from running.xgboost.xgb_model import XGBModel
import os
import xgboost as xgb
from util.data_process import DataProcess as dtp
from util import evaluate_process as elp
from util import visualization
from util import output_process as opp

"""
原生的xgboost 训练
"""

project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

if __name__ == '__main__':
    with open(f'{project_path}/running/xgboost/car_price_feat.txt') as f:
        feat_list = list(filter(lambda line: line[0] != '#', f.read().split('\n')))
    ###
    general_params = {
        'booster': 'gbtree',  # gbtree, gblinear, dart
        'silent': 0,  #
        'verbosity': 2,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        # 'nthread': 2,
        'disable_default_eval_metric': 0,  # Set to >0 to disable.
    }
    tree_params = {
        'eta': 0.055,  # learning_rate, default 0.3
        'gamma': 0.2,  # min_split_loss  越大越保守
        'max_depth': 3,
        'min_child_weight': 3,  # 越大越保守, 叶子节点上所有样本的权重和小于min_child_weight则停止分裂
        # 'max_delta_step': 0,  # it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 1,
        'colsample_bynode': 1,
        'lambda': 1,  # L2
        'alpha': 0.3,
        'tree_method': 'hist',  # [auto, exact, approx, hist, gpu_hist]
        # 'sketch_eps': 0.03,  # Only used for tree_method=approx.
        # 'scale_pos_weight': 1,  # sum(negative instances) / sum(positive instances)
        'grow_policy': 'lossguide',  # [depthwise, lossguide] only if tree_method is set to hist
        'max_leaves': 0,  # Only relevant when grow_policy=lossguide is set
        'max_bin': 1024,  # Only used if tree_method is set to hist. 直方图数目, 越大越容易过拟合
    }
    task_params = {
        'objective': 'reg:linear',  # 'reg:linear',
        'eval_metric': 'rmse',  # rmse for regression, and error for classification, mean average precision for ranking

    }
    xgb_params = {**general_params, **tree_params, **task_params}
    ###
    data_train = pd.read_csv(f'{project_path}/data/car_price_train.201908.csv')
    data_test = pd.read_csv(f'{project_path}/data/car_price_test.201908.csv')
    ###
    label_encode_map, f_map = dtp.gencode(pd.concat([data_train, data_test]), feat_list)
    ####
    series_name = '宝马5系'
    d_train = data_train[data_train.model_series == series_name]
    d_test = data_test[data_test.model_series == series_name]
    print(len(d_train), len(d_test))
    en_train = dtp.encode_process(d_train, feat_list, label_encode_map)
    en_test = dtp.encode_process(d_test, feat_list, label_encode_map)
    ####
    train_df = xgb.DMatrix(en_train[feat_list], label=en_train.price)
    test_df = xgb.DMatrix(en_test[feat_list], label=en_test.price)
    ###
    evals = [(train_df, 'train'), (test_df, 'test')]
    bst = xgb.train(xgb_params, train_df, 215, evals=evals)
    ###
    pred = bst.predict(test_df)
    elp.car_price_evaluate(d_test, pred)
    ###
    vdf = d_test[['price', 'id']].copy()
    vdf['pred'] = pred
    vdf = vdf.sort_values(by='price')
    visualization.gen_line_plotly('./output/car_price_proto.html', df={'price': vdf.price, 'pred': vdf.pred},
                                  attrs=vdf.id,
                                  consult_cols=[vdf.price * .95, vdf.price * 1.05],
                                  auto_open=1,
                                  graph_name=series_name
                                  )
