import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath('.')), '../'))
import sys

sys.path.append(project_path)
from util.visualization import draw_lines
from util.visualization import draw_scatters
from util.evaluate_process import classifier_evaluate
from collections import Counter
from sklearn.metrics import auc, roc_curve
from numpy import mat, zeros, inf, ones, log, sign, multiply, dot, exp


## 单层决策树作为基分类器
def stump_classify(df, dimen, thresh_val, condition):  # just classify the data
    vals = df[dimen]
    if condition == 'lt':
        return [1 if x <= thresh_val else -1 for x in vals]
    else:
        return [1 if x >  thresh_val else -1 for x in vals]


def build_stump(df, labels, D=None):
    """
    D 样本权重向量
    """
    m, n = df.shape
    if D is None:
        D = np.array([1 / m] * m)
    n_steps = 10.0  # 特征值分成10份, 如果是离散值就先判断离散值的数量
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity
    for feat in df.columns:
        # loop over all dimensions
        # 遍历所有特征
        uniq_vals = set(df[feat])
        if len(uniq_vals) > n_steps:
            rangeMin = min(df[feat])
            rangeMax = max(df[feat])
            stepSize = (rangeMax - rangeMin) / n_steps
            thresh_vals = [rangeMin + k * stepSize for k in range(int(n_steps) + 1)]
        else:
            thresh_vals = uniq_vals
        for thresh_val in thresh_vals:  # loop over all range in current dimension
            for condition in ['lt', 'gt']:  # go over less than and greater than
                predictedVals = stump_classify(df, feat, thresh_val,
                                               condition)  # call stump classify with i, j, lessThan
                errArr = [0 if p == l else 1 for p, l in zip(predictedVals, labels)]
                weightedError = np.dot(D.T, errArr)  # 加权误差率
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = feat
                    bestStump['thresh'] = thresh_val
                    bestStump['condition'] = condition
    return bestStump, minError, bestClasEst


def cal_sample_weight(row, alpha):
    expon = -alpha * row.label * row.label_est
    c = np.exp(expon)
    return row.sample_weight * c


def adaboost_fit(df, labels, n_estimators=40, random_state=None):
    base_classifiers = []
    m = len(df)
    rdf = pd.DataFrame({
        'label': labels,
        'label_est': None,
        'sample_weight':  [1/m] * m,  # 样本权重
        'agg_label_est': [0] * m
    })

    for i in range(n_estimators):
        best_stump, error, label_est = build_stump(df, labels, rdf.sample_weight)  # 调用基分类器
        alpha = 0.5 * np.log((1.0 - error) / error)  # 基分类器权重
        best_stump['alpha'] = alpha
        base_classifiers.append(best_stump)  # store Stump Params in Array
        #### 计算新的样本权重分布
        rdf['label_est'] = label_est
        # w => 𝑤(𝑚,𝑖)𝑒𝑥𝑝(−α𝑚𝑦𝑖𝐺𝑚(𝑥𝑖)) w 是个中间结果
        w = rdf.sample_weight * np.exp(-alpha * rdf.label * rdf.label_est)
        z = sum(w)
        rdf['sample_weight'] = w/z
        # 已有模型的加权结果
        rdf['agg_label_est'] = rdf.agg_label_est + alpha * rdf.label_est
        agg_errors = len(rdf[rdf.agg_label_est.map(np.sign) != rdf.label])
        agg_error_rate = agg_errors / m
        print(i, "total error: ", agg_error_rate)
        if agg_error_rate == 0.0:
            break
    return base_classifiers


def adaboost_predict(df, base_models):
    m = len(df)
    rdf = pd.DataFrame({'agg_label_est': [0] * m})
    for clf in base_models:
        rdf['label_est'] = stump_classify(df, clf['dim'], clf['thresh'], clf['condition'])
        rdf['agg_label_est'] = rdf.apply(lambda row: row.agg_label_est + clf['alpha'] * row.label_est, axis=1)
    return sign(rdf.agg_label_est)


if __name__ == '__main__':
    df = pd.read_csv('../../data/preprocessed.samecar.csv')
    feats = [
        'colorp1', 'colorp2',
        'fuel_typep1', 'fuel_typep2', 'displacement_standard1', 'displacement_standard2',
        'gearboxp1', 'gearboxp2', 'displacement_diff', 'displacement_diff_sparse',
        'mile_diff', 'mile_diff_sparse', 'mile_diff_rate', 'mile_diff_rate_sparse',
        'year_diff', 'year_diff_sparse', 'licensed_city_diff_sparse', 'title_diff',
        'title_diff_sparse', 'register_time_diff', 'register_time_diff_sparse',
        'is_import_diff_sparse', 'transfer_times_diff', 'transfer_times_diff_sparse'
    ]
    # 只保留离散特征
    sparse_feats = [
        'colorp1', 'colorp2',
        'fuel_typep1', 'fuel_typep2', 'displacement_standard1', 'displacement_standard2',
        'gearboxp1', 'gearboxp2', 'displacement_diff_sparse',
        'mile_diff_sparse', 'mile_diff_rate_sparse',
        'year_diff_sparse', 'licensed_city_diff_sparse',
        'title_diff_sparse', 'register_time_diff_sparse',
        'is_import_diff_sparse', 'transfer_times_diff_sparse'
    ]
    df['is_same'] = df.is_same.map(lambda x: -1 if x == 0 else 1)
    rdf = df[feats]
    X_train, X_test, y_train, y_test = train_test_split(rdf, df.is_same, test_size=0.25, random_state=10)
    models = adaboost_fit(X_train, y_train, n_estimators=20)
    pred = adaboost_predict(X_test, models)
    print(classifier_evaluate(y_test, pred, [1, -1]))