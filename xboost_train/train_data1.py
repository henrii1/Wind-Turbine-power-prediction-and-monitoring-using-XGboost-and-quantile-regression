import pandas as pd
import numpy as np
import time
from sklearn.metrics import *
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from loggings import logger

data_dir = 'https://github.com/henrii1/Wind-Turbine-power-prediction-and-monitoring-using-XGboost-and-quantile-regression/blob/main/data/kelmarsh_02.xlsx?raw=true'
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
        F = 0.95
    elif ((x > 3) and (x <= 10)):
        F = 0.9
    elif ((x > 10) and (x <= 20)):
        F = 0.92
    elif ((x > 20) and (x <= 30)):
        F = 0.96
    else:
        F = 0.985
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

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 1)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    print(f'RMSE:{rmse}')
    print(f'MAE:{mae}')
    print(f'R_SCORE:{r2}')

best_xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)


best_xgb_model.fit(x_train,y_train)
pred1 = best_xgb_model.predict(x_test)

eval_metrics(y_test, pred1)
