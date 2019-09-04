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


def car_price_evaluate(d_test, pred):
    new_d_test = d_test.copy()
    new_d_test['pred'] = pred
    new_d_test['err'] = new_d_test.apply(lambda row: row.price - row.pred, axis=1)
    new_d_test['abserr'] = new_d_test.err.map(lambda x: abs(x))
    new_d_test['ape'] = new_d_test.apply(lambda row: row.abserr / row.price, axis=1)
    mape = np.mean(new_d_test.ape)
    print(f'MAPE: {round(mape * 100, 2)}%')
    accuracy5p = len(list(filter(lambda x: x <= 0.05, new_d_test.ape))) / len(new_d_test)
    print(f'5%: {round(accuracy5p, 4) * 100}%')
    msev = float(np.mean(new_d_test.err * new_d_test.err))
    print(f'mse: {round(msev, 4)}')
    rmse = np.sqrt(msev)
    print(f'rmse: {round(rmse, 4)}')


def classifier_evaluate(target, pred):
    df = pd.DataFrame({'target': target, 'pred': pred})
    recall = len(df[(df.target == 1) & (df.pred == 1)]) / len(df[df.target == 1])
    precision = len(df[(df.target == 1) & (df.pred == 1)]) / len(df[df.pred == 1])
    accuracy = len(df[df.target == df.pred]) / len(df)
    ret = {
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'tp': len(df[(df.target == 1) & (df.pred == 1)]),
        'fp': len(df[(df.target == 0) & (df.pred == 1)]),
        'tn': len(df[(df.target == 0) & (df.pred == 0)]),
        'fn': len(df[(df.target == 1) & (df.pred == 0)])
    }
    return ret
