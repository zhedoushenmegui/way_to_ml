import pandas as pd
import os


data_train = pd.read_csv(f'../data/car_price_train.201908.csv')
data_test = pd.read_csv(f'../data/car_price_test.201908.csv')
###
series_name = '宝马5系'
d_train = data_train[data_train.model_series == series_name]
d_test = data_test[data_test.model_series == series_name]


d_train.to_csv('small_car_price_train.201908.csv', index=None)
d_test.to_csv('small_car_price_test.201908.csv', index=None)