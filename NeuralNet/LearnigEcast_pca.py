from EcastModel import EacstNetwork

import pandas as pd


learningModel = EacstNetwork(20, 127, 1e-2, 'denchu')

df = pd.read_csv('../DL_ECast/Data/191217_dataset.csv', sep=',')
x = df.iloc[:, 1:42]
y = df.iloc[:, 44]

learningModel.run(x, y, 0, 0.2)
