import pandas as pd
import keras
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA


# return: DataFrame
def load(filepath):
    df = pd.read_csv(filepath, sep=',')
    return df


# df: DataFrame
# targetcol: df['col2']
def convertOneHot(df, targetCol):
    result = pd.get_dummies(df, columns=[targetCol], drop_first=True)
    return result


# x: DataFrame
# return: ndarray
def standalize(x):
    sc = preprocessing.StandardScaler()
    sc.fit(x)
    x_std = sc.transform(x)
    return x_std


# x: ndarray
# return: nparray
def calcPCA(x, dimensions=0):
    if dimensions == 0:
        pca = PCA()
    else:
        pca = PCA(n_components=dimensions)

    pca.fit(x)
    PCA(copy=True, whiten=False)
    transformdata = pca.fit_transform(x)

    return transformdata
