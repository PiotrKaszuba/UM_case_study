import pandas as pd
from sklearn.preprocessing import MinMaxScaler,scale
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error, me
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import dataPrepare as dp


# komentarze prezentują próby poprawy wyniku - nieudane LUB wykonane wcześniej dobory parametrów/cech

def createModel(k):
    return neighbors.KNeighborsRegressor(n_neighbors =k,algorithm='kd_tree', n_jobs=-1)



dfx = dp.getData(100000)
target = dp.target

from FeatureEngineeringManualRun import getFeatures

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




# dla automatycznej selekcji - ('killPlace', 'kills', 'maxPlace', 'numGroups', 'walkDistance')
#df = dfx[['winPlacePerc', 'walkDistance', 'killPlace', 'numGroups', 'maxPlace', 'kills']]
#[9] - okazuje się, że zachowanie numGroups poprawia wyniki
#[8] - odrzucenie zmiennych grupowych poprawia wyniki

# wszystkie numeryczne - score 0.1002
#df = dfx.drop(columns=['Id', 'groupId', 'matchId', 'matchType'])

# log transform [2]

#brak - 0.08526

# wsp. skośności > 2 (kills) -  0.08531

# najlepsza kolumna - walkDistance - 0.07549 -> !! mały wsp. skośności -> normalizacja sprawia, że nie opłaca się tego robić
# skewed = ['walkDistance']
# for skew in skewed:
#     df[skew] = np.log(df[skew]+1)


# #[5] walkDistance features  !! po normalizacji
# # brak - 0.06754
# #
# # # czy więcej od zera -> 0.06757
# df['walkMoreThan0'] = dfx['walkDistance'] > 0
# df['walkMoreThan0'] = df['walkMoreThan0'].astype(int)
# #
# # # czy więcej od pierwszego kwartyla > 155.1 -> 0.06776
# df['walkMoreThanQ1'] = dfx['walkDistance'] > 155.1
# df['walkMoreThanQ1'] = df['walkMoreThanQ1'].astype(int)
# #
# # # czy więcej od mediany  > 685.6 -> 0.06769
# df['walkMoreThanQ2'] = dfx['walkDistance'] > 685.6
# df['walkMoreThanQ2'] = df['walkMoreThanQ2'].astype(int)
#
# # czy więcej od trzeciego kwartyla  > 1976.00 -> 0.06765
# df['walkMoreThanQ3'] = dfx['walkDistance'] > 1976.00
# df['walkMoreThanQ3'] = df['walkMoreThanQ3'].astype(int)
#
#
# # czy więcej od średniej > 1154.22 -> 0.06769
# df['walkMoreThanQ1'] = dfx['walkDistance'] > 1154.22
# df['walkMoreThanQ1'] = df['walkMoreThanQ1'].astype(int)


# przy odrzuceniu kolumny walkDistance zestaw tych wartości
# poprawia wynik z 0.084 na 0.070
# df = df.drop(columns=['walkDistance'])

# [6] weaponsAcquired !! po normalizacji
# brak - 0.06754

# #próba dodania faktu znalezienia conajmniej jednej broni -> 0.06786
# df['weaponsAcquiredMoreZero'] = dfx['weaponsAcquired'] > 0
# df['weaponsAcquiredMoreZero'] = df['weaponsAcquiredMoreZero'].astype(int)
#
# #próba dodania faktu znalezienia conajmniej dwóch broni -> 0.06811
# df['weaponsAcquiredMore1'] = dfx['weaponsAcquired'] > 1
# df['weaponsAcquiredMore1'] = df['weaponsAcquiredMore1'].astype(int)
#
# #próba dodania faktu znalezienia conajmniej 4 broni -> 0.06852
# df['weaponsAcquiredMore3'] = dfx['weaponsAcquired'] > 3
# df['weaponsAcquiredMore3'] = df['weaponsAcquiredMore3'].astype(int)

#próba dodania faktu znalezienia conajmniej 10 broni -> 0.06785
# df['weaponsAcquiredMore9'] = dfx['weaponsAcquired'] > 9
# df['weaponsAcquiredMore9'] = df['weaponsAcquiredMore9'].astype(int)

#[4] próba dołączenia kolumy rankPoints - zmienionej !! po normalizacji
#rankPoints -> zastąpienie średnią wartości [-1, 0]
# brak - 0.06754

# dodanie zmiennej = 1 jeśli rankPoints to -1 lub 0
# dodanie -> 0.06875
# df['rankPointsZero'] = dfx['rankPoints'] <= 0
# df['rankPointsZero'] = df['rankPointsZero'].astype(int)
# dodanie -> 0.07133
# rp = dfx['rankPoints']
# rpMean = rp[rp>0].mean()
# rp=rp.replace([-1,0], rpMean)
# df['rankPoints'] = rp
# obie -> 0.07234

# dosunięcie wartości -> 0.06900
# rp = dfx['rankPoints']
# rpMin = rp[rp>0].min()
# print(rpMin)
# rp=rp.replace([0], rpMin-1)
# rp=rp.replace([-1], rpMin-2)
# df['rankPoints'] = rp

# dodanie rankPoints bez zmian -> 0.06898
# df['rankPoints'] = dfx['rankPoints']


print(df.head())


#x_train, y_train, x_test, y_test = dp.get_train_valid_sets(df)
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


#[11] - PCA
# brak - 0.06754
# dla 5->4 wymiary 0.06773
# potem gorzej

# pca = PCA(n_components=len(df.columns)-1)
# x_train = pca.fit_transform(x_train)
# x_train = pd.DataFrame(data = x_train)
# x_test = pca.transform(x_test)
# x_test = pd.DataFrame(data = x_test)


# dobieranie k -> było wybierane najlepsze z 1-30 w doborze parametrów
# dla już dobranych parametrów najlepsze k wynosi 19 dla 50k próbek
#oszacowanie najlepszego k w zależności od n (n->k)
#300k -> 40 ; 0.06620
#200k -> 32 ; score - 0.06646

# do 50k wydawało się, że a*log2(n) będzie dobrą funkcją
#50k -> 19
#12,5k -> 14
#3k -> 10
#750 -> 7
#200 -> 4





# minK = 0
# maxK = 30
# rmse_val = [] #to store rmse values for different k
# for K in range(minK,maxK):
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
# b=pd.Series(list(range(minK,maxK)))
# a.name='a'
# b.name='b'
# curve = pd.concat([a,b], axis=1) #elbow curve
# curve.plot(x='b')
# print("Best is {0}, at index {1}".format(min(rmse_val), np.argmin(rmse_val)+1))
# plt.show()
#best_k = np.argmin(rmse_val)+1
best_k=19 ## wystrojone już
#Ensemble, different k -> 0.06736
model = neighbors.KNeighborsRegressor(n_neighbors = best_k,algorithm='kd_tree' )
model2 = neighbors.KNeighborsRegressor(n_neighbors = int(best_k/2),algorithm='kd_tree' )
model3 = neighbors.KNeighborsRegressor(n_neighbors = best_k*2,algorithm='kd_tree' )
model4 = neighbors.KNeighborsRegressor(n_neighbors = best_k-2,algorithm='kd_tree')
model5 = neighbors.KNeighborsRegressor(n_neighbors = best_k+2,algorithm='kd_tree')
ensemble = VotingRegressor([('m1',model), ('m2',model2), ('m3',model3), ('m4',model4), ('m5',model5)], weights=[1,1,1,1,1])
ensemble.fit(x_train, y_train)
pred=ensemble.predict(x_test) #make prediction on test set
error = mean_absolute_error(y_test,pred) #calculate err

#error2 = np.sqrt(mean_squared_error(y_test,pred)) #calculate err2
print('MAE: ', error)
#print('RMSE: ', error)