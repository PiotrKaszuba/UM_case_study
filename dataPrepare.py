import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split

categorical_cols = ['Id', 'groupId', 'matchId', 'matchType']
target='winPlacePerc'

def getValidData(size=None):
    dfx = pd.read_csv("trainSmall1.csv")
    dfx = dfx.dropna()
    if size is not None:
        dfx = dfx.iloc[210000:210000+size]
    return dfx

def getTestData(size=None):
    dfx = pd.read_csv("localTest.csv")
    dfx = dfx.dropna()
    if size is not None:
        dfx = dfx.head(size)#dfx = dfx.iloc[210000:210000+size]
    return dfx
def getData(size=None):
    dfx = pd.read_csv("trainSmall1.csv")
    dfx = dfx.dropna()
    if size is not None:
        dfx = dfx.head(size)
    return dfx

def bestFeatures(model, df, cols_drop=None, preprocess_f=None):
    if cols_drop is not None:
        df = df.drop(columns=cols_drop)
    X = df.drop(columns=target)
    y = df[target]

    if preprocess_f is not None:
        X = preprocess_f(X)
    max_feats = len(X.columns)

    sfs1 = SFS(model,
               k_features=(4,13),
               forward=True,
               floating=False,
               verbose=2,
               scoring='neg_mean_absolute_error',
               cv=4,
               n_jobs=8)
    sfs1 = sfs1.fit(X, y)
    print(sfs1.k_feature_names_)
    return list(sfs1.k_feature_names_)

def standardize(df, mean=None, std = None, ret_mean_std = False):
    if mean is None:
        mean = df.mean()
    if std is None:
        std = df.std()
    df = (df-mean)/(std+0.0001)
    df = df.fillna(value=0.0)
    if ret_mean_std:
        return df, mean, std
    else:
        return df

def get_train_valid_sets(df, size=0.3):
    train, test = train_test_split(df, test_size=size, random_state=1)

    x_train = train.drop(target, axis=1)
    y_train = train[target]

    x_test = test.drop(target, axis=1)
    y_test = test[target]

    return x_train, y_train, x_test, y_test