from sklearn.linear_model import LinearRegression

import pandas as pd
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import dataPrepare as dp
from FeatureEngineeringManualRun import getFeatures

lr = LinearRegression(normalize=False)

dfx = dp.getData(200000)
target = dp.target



dfx = getFeatures(dfx)

testDf = dp.getValidData(30000)
testDf = getFeatures(testDf)
df = dfx.drop(columns=dp.categorical_cols)
features = list(df.columns)
features.remove(dp.target)
x_train = df[features]
y_train = df[target]
x_test = testDf[features]
y_test = testDf[target]



#
#
for column in x_train.columns:
    x_train[column+'log'] = np.log(x_train[column]+1.1)
    x_test[column+'log'] = np.log(x_test[column]+1.1)

x_train, mean, std = dp.standardize(x_train, ret_mean_std=True)
x_test = dp.standardize(x_test, mean, std)
#x_test = x_test.dropna()
pca = PCA(n_components=28)
x_train = pca.fit_transform(x_train)
x_train = pd.DataFrame(data = x_train)
x_test = pca.transform(x_test)
x_test = pd.DataFrame(data = x_test)
print(x_train.columns)
# pca = PCA().fit(x_train)
# #(https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe)
# #Plotting the Cumulative Summation of the Explained Variance
# plt.figure()
# plt.plot(np.cumsum([0] + pca.explained_variance_ratio_.tolist()))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title('Explained Variance')
# plt.scatter(list(range(1,len(x_train.columns)+1,3)), np.cumsum(pca.explained_variance_ratio_)[0:len(x_train.columns):3])
# for i in range(0,len(x_train.columns),3):
#     plt.annotate("{0:.2f}".format(np.cumsum(pca.explained_variance_ratio_)[i]), (i+1, np.cumsum(pca.explained_variance_ratio_)[i]))
#     plt.axvline(x=i+1)
# #
# plt.xticks( list(range(1,len(x_train.columns)+1,3)))
# plt.show()
poly = PolynomialFeatures(degree=3)
x_train = poly.fit_transform(x_train)
x_test = pd.DataFrame(poly.fit_transform(x_test))







lr.fit(x_train, y_train)
print(lr.coef_)
print(len(lr.coef_))
pred=lr.predict(x_test)
error = mean_absolute_error(y_test,pred) #calculate err
r2 = r2_score(y_test, pred)

AdjR2 = 1-(1-r2)*(200000-1)/(200000-len(lr.coef_)-1)
print('AdjR2: ', AdjR2)
print('MAE: ', error)
print('R2: ', r2)