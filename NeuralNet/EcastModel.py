from NNModel import NeuralNetwork

import keras
from keras.layers import Dense, Dropout
from NNSubModule import BeforeProcess
from keras.optimizers import Adam


class EacstNetwork(NeuralNetwork):
    def PreProcessing(self, rawdata, param):
        data_onehot = BeforeProcess.convertOneHot(rawdata, 'item_1')
        data_std = BeforeProcess.standalize(data_onehot)
        data_pca = BeforeProcess.calcPCA(data_std)
        return data_pca

    def InitializeModel(self, n):
        self.model.add(Dense(40, input_dim=n, activation='relu', kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(0.1)))
        self.model.add(
            Dense(30, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.1)))
        self.model.add(
            Dense(20, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.1)))
        self.model.add(
            Dense(10, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.1)))
        self.model.add(
            Dense(5, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.1)))
        self.model.add(Dense(1))

        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-2), metrics=['accuracy'])

    def saveResults(self):
        pass
