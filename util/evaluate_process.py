import pandas as pd
import numpy as np

"""
评估相关的方法
"""


def mse(target, pred):
    return np.mean([(p-t)**2 for p, t in zip(pred, target)])


def rmse(target, pred):
    return np.mean([(p-t)**2 for p, t in zip(pred, target)]) ** 0.5


def mae(target, pred):
    return np.mean([abs(p-t) for p, t in zip(pred, target)])


def car_price_evaluate(target, pred, silent=0):
    if isinstance(target, pd.DataFrame):
        new_d_test = target.copy()
        new_d_test['pred'] = pred
    else:
        new_d_test = pd.DataFrame({'price': target, 'pred':pred})

    new_d_test['err'] = new_d_test.apply(lambda row: row.price - row.pred, axis=1)
    new_d_test['abserr'] = new_d_test.err.map(lambda x: abs(x))
    new_d_test['ape'] = new_d_test.apply(lambda row: row.abserr / row.price, axis=1)
    mape = np.mean(new_d_test.ape)
    accuracy5p = len(list(filter(lambda x: x <= 0.05, new_d_test.ape))) / len(new_d_test)
    msev = float(np.mean(new_d_test.err * new_d_test.err))
    rmse = np.sqrt(msev)
    if not silent:
        print(f'rmse: {round(rmse, 4)}')
        print(f'mse: {round(msev, 4)}')
        print(f'MAPE: {round(mape * 100, 2)}%')
        print(f'5%: {round(accuracy5p, 4) * 100}%')
    return {
        'rmse': round(rmse, 4),
        'mse': round(msev, 4),
        'mape': round(mape*100, 4),
        '5%': round(accuracy5p, 4) * 100
    }


def classifier_evaluate(target, pred, labels=None):
    """
    二分类模型评估
    :param labels:
    :param target:
    :param pred:
    :return:
    """
    if labels is None:
        positive, negative = 1, 0
    else:
        positive, negative = labels[0], labels[1]
    df = pd.DataFrame({'target': list(target), 'pred': list(pred)})
    recall = len(df[(df.target == positive) & (df.pred == positive)]) / len(df[df.target == positive])
    precision = len(df[(df.target == positive) & (df.pred == positive)]) / len(df[df.pred == positive])
    accuracy = len(df[df.target == df.pred]) / len(df)
    f1score = 2 * recall * accuracy / (recall + accuracy)
    ret = {
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1score,
        'tp': len(df[(df.target == positive) & (df.pred == positive)]),
        'fp': len(df[(df.target == negative) & (df.pred == positive)]),
        'tn': len(df[(df.target == negative) & (df.pred == negative)]),
        'fn': len(df[(df.target == positive) & (df.pred == negative)])
    }
    return ret
