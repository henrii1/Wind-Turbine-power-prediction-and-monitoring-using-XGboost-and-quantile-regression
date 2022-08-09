import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Flatten, Layer
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from loggings import logger

data_dir = 'https://github.com/henrii1/Wind-Turbine-power-prediction-and-monitoring-using-XGboost-and-quantile-regression/blob/main/data/penmanshiel_14.xlsx?raw=true'
data = pd.read_excel(data_dir)
data.isna().sum()

def outlier_remover(dat, prop, min, max):
    d = dat
    q_low = d[prop].quantile(min)
    q_high = d[prop].quantile(max)
    return d[(d[prop]<q_high) & (d[prop]>q_low)]

d1 = {}
step = 50
i = 1
for x in range(20, 3100, step):
    d1[i] = data.iloc[((data['power']>=x)& (data['power']<x+step)).values]
    i = i + 1

d1[-2] = data.iloc[(data['power']>=2900).values]

for x in range(1, 62):
    if x <= 3:
        F = 0.96
    elif ((x > 3) and (x <= 6)):
        F = 0.84
    elif ((x > 6) and (x <= 10)):
        F = 0.93
    elif ((x > 10) and (x <= 13)):
        F = 0.95
    elif ((x > 13) and (x <= 20)):
        F = 0.90
    elif ((x > 20) and (x < 30)):
        F = 0.88
    else:
        F = 0.9
    d1[x] = outlier_remover(d1[x], 'wind speed', 0.00001, F)


df = pd.DataFrame()
for infile in range (1, 62):
    data = d1[infile]
    df = df.append(data, ignore_index = True)

da = df.drop(columns=['date'])
da.dropna()
scaler = MinMaxScaler(feature_range =(0, 1))
data_ = scaler.fit_transform(da)
data_x = data_[:, :-1]
data_y = data_[:, -1]

x_train_, x_test, y_train_, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 1)
x_train, x_test, y_train, y_test = train_test_split(x_train_, y_train_, test_size = 0.25, random_state = 2)