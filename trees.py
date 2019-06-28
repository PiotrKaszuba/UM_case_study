from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import dataPrepare as dp



dfx = dp.getData()
target = dp.target
#df = dfx.drop(columns=['Id', 'groupId', 'matchId', 'matchType'])


def createModel(k):
    return GradientBoostingRegressor(n_estimators=k)

features = dp.bestFeatures(createModel(20), dfx, dp.categorical_cols, dp.standardize)


df = dfx[features]
df[target] = dfx[target]

x_train, y_train, x_test, y_test = dp.get_train_valid_sets(df)

x_train, mean, std = dp.standardize(x_train, ret_mean_std=True)
x_test = dp.standardize(x_test, mean, std)


clf = GradientBoostingRegressor(n_estimators=200)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
error = mean_absolute_error(y_test,y_pred) #calculate err
print('MAE value  is:', error)