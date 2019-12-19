import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('seiseki.csv')

pca = PCA()
feature = pca.fit(df)

feature = pca.transform(df)

plt.figure(figsize=(6,6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

print(pca.explained_variance_ratio_)
