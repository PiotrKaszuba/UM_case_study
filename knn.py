import pandas as pd
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import dataPrepare as dp
from FeatureEngineeringManualRun import getFeatures

# komentarze prezentują próby poprawy wyniku - nieudane LUB wykonane wcześniej dobory parametrów/cech

def createModel(k):
    return neighbors.KNeighborsRegressor(n_neighbors =k,algorithm='kd_tree', n_jobs=-1)



dfx = dp.getData(200000)
target = dp.target



dfx = getFeatures(dfx)

testDf = dp.getTestData(30000)
testDf = getFeatures(testDf)

# #k =15 działa dobrze dla tych wartości n
#features = dp.bestFeatures(createModel(15), dfx, dp.categorical_cols, dp.standardize)
#print(features)
#features = list(('killPlace', 'kills', 'maxPlace', 'numGroups', 'walkDistance'))
#best chosen features
features = ['killPlace', 'kills', 'killStreaks', 'matchDuration', 'maxPlace', 'numGroups', 'f1_5', 'f1_8', 'TeamWalkDistanceMean']
# wybór cech [10]:
# najlepsze - score 0.08526
# dobrane metodą forward selection
df = dfx[features]
df[target] = dfx[target]

testDftarget = testDf[target]
testDf = testDf[features]
testDf[target] = testDftarget

print(df.head())

# x_train, y_train, x_test, y_test = dp.get_train_valid_sets(df, size=0.1304347826)
x_train = df[features]
y_train = df[target]
x_test = testDf[features]
y_test = testDf[target]


# [1] - skalowanie - normalizacja
#brak - 0.07549 -> z log_transform na walkDistance

#MinMax - 0.06949 best score -> bez log_transform
# scaler = MinMaxScaler()
# x_train=pd.DataFrame(scaler.fit_transform(x_train))
# x_test=pd.DataFrame( scaler.transform(x_test))

# mean/std - 0.06754 best score, -=-
x_train, mean, std = dp.standardize(x_train, ret_mean_std=True)
x_test = dp.standardize(x_test, mean, std)
#
# pca = PCA(n_components=len(df.columns)-1)
# x_train = pca.fit_transform(x_train)
# x_train = pd.DataFrame(data = x_train)
# x_test = pca.transform(x_test)
# x_test = pd.DataFrame(data = x_test)
# print(x_train.columns)
# pca = PCA().fit(x_train)
# #(https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe)
# #Plotting the Cumulative Summation of the Explained Variance
# plt.figure()
# plt.plot(np.cumsum([0] + pca.explained_variance_ratio_.tolist()))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title('Explained Variance')
# plt.scatter(list(range(1,10,1)), np.cumsum(pca.explained_variance_ratio_)[0:9])
# for i in range(0,9,1):
#     plt.annotate("{0:.3f}".format(np.cumsum(pca.explained_variance_ratio_)[i]), (i+1, np.cumsum(pca.explained_variance_ratio_)[i]))
#     plt.axvline(x=i+1)
#
# plt.xticks( list(range(1,10,1)))
# plt.show()

# dobieranie k -> było wybierane najlepsze z 1-30 w doborze parametrów
# dla już dobranych parametrów najlepsze k wynosi 19 dla 50k próbek
#oszacowanie najlepszego k w zależności od n (n->k)
# wydaje się, że a*log2(n) będzie dobrą funkcją
#200k -> 22
#50k -> 19
#12,5k -> 14
#3k -> 10
#750 -> 7
#200 -> 4

# minK = 1
# maxK = 30
# rmse_val = [] #to store rmse values for different k
# for K in range(minK,maxK,4):
#     K = K+1
#     # ważenie przy obliczaniu średniej pogarsza wynik  0.06754 -> 0.06785
#     # zmiana miary odległości również zaszkodziła
#     model = neighbors.KNeighborsRegressor(n_neighbors = K,algorithm='kd_tree', n_jobs=-1)
#     print(K)
#     model.fit(x_train, y_train)  #fit the model
#     print(K)
#     pred=model.predict(x_test) #make prediction on test set
#     error = mean_absolute_error(y_test,pred) #calculate err
#     rmse_val.append(error) #store err values
#     print('MAE value for k= ' , K , 'is:', error)
#
# a=pd.Series(rmse_val)
# b=pd.Series(list(range(minK,maxK, 4)))
# a.name='a'
# b.name='b'
# curve = pd.concat([a,b], axis=1) #elbow curve
# curve.plot(x='b')
# print("Best is {0}, at index {1}".format(min(rmse_val), np.argmin(rmse_val)+1))
# plt.show()
# best_k = np.argmin(rmse_val)+1
best_k=22 ## wystrojone już


#Ensemble, different k -> 0.06736

## final setup without ensemble-> 0.063695 , with -> 0.0635..
model = neighbors.KNeighborsRegressor(n_neighbors = best_k,algorithm='kd_tree', weights='distance' )
model2 = neighbors.KNeighborsRegressor(n_neighbors = int(best_k/2),algorithm='kd_tree', weights='distance' )
model3 = neighbors.KNeighborsRegressor(n_neighbors = best_k*2,algorithm='kd_tree' , weights='distance')
model4 = neighbors.KNeighborsRegressor(n_neighbors = best_k-2,algorithm='kd_tree', weights='distance')
model5 = neighbors.KNeighborsRegressor(n_neighbors = best_k+2,algorithm='kd_tree', weights='distance')
ensemble = VotingRegressor([('m1',model), ('m2',model2), ('m3',model3), ('m4',model4), ('m5',model5)], weights=[1,1,1,1,1])
ensemble.fit(x_train, y_train)
# model.fit(x_train, y_train)
pred=ensemble.predict(x_test) #make prediction on test set
error = mean_absolute_error(y_test,pred) #calculate err
r2 = r2_score(y_test, pred)
print('MAE: ', error)
print('R2: ', r2)