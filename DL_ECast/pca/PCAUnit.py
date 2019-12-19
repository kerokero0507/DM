from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

"""import pandas as pd
X = pd.read_csv('/home/mizuguchi/repos/DM/DL_ECast/Data/191217_dataset.csv')

Y = pca.fit(X)

PCA(copy=True, whiten=False)

#print(pca.components_)
#print(pca.mean_)
#print(pca.get_covariance())

print(pca.explained_variance_ratio_)
import numpy
print(numpy.sum(pca.explained_variance_ratio_)) """

class pcaExchange:
    def __init__(self, ):
        self.x = 0

    def exchange(self, data):
        # print(data)
        pca = PCA()
        pca.fit(data)
        PCA(copy=True, whiten=False)

        transformdata = pca.fit_transform(data)

        # print(transformdata)
        
        # print(pca.explained_variance_ratio_)

        # x = np.arange(1, 43, 1)
        # print(x)
        # plt.bar(x, pca.explained_variance_ratio_)
        # plt.show()

        return transformdata
