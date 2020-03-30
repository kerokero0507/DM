import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split

from NNSubModule import BeforeProcess


class NeuralNetwork:
    def __init__(self, epochs, batchSize, leaningRate, modelName):
        self.epoch = epochs
        self.size = batchSize
        self.rate = leaningRate
        self.name = modelName

        self.model = Sequential()

    def run(self, rawData, labelData, param, testSize):
        # beforeProcess
        treatmentData = self.PreProcessing(rawData, param)
        # -------------
        histryResut, scoreResut = self.Learning(treatmentData, labelData, testSize, True)

        return histryResut, scoreResut

    def Learning(self, teachData, labelData, testSize, isSaveResults):
        self.InitializeModel(teachData.shape[1])

        x_train, x_test, t_train, t_test = train_test_split(teachData, labelData, test_size=testSize, random_state=0)

        history = self.model.fit(x_train, t_train, batch_size=self.size, epochs=self.epoch, verbose=1,
                                 validation_data=(x_test, t_test))
        score = self.model.evaluate(x_test, t_test, verbose=0)

        if isSaveResults:
            pred_train = self.Predict(x_train)
            # pred_train.to_csv(self.name + '_pred_train.csv', index=False)
            np.savetxt(self.name + '_pred_train.csv', pred_train)

            pred_test = self.Predict(x_test)
            # pred_test.to_csv(self.name + '_pred_test.csv', index=False)
            np.savetxt(self.name + '_pred_test.csv', pred_test)

            t_train.to_csv(self.name + '_label_train.csv', index=False)
            t_test.to_csv(self.name + '_label_test.csv', index=False)

        return history, score

    def Predict(self, inputData):
        result = self.model.predict(inputData)
        return result

    def Save(self, filePath):
        # filename extension : xxxx.h5
        self.model.save(filePath)

    def Load(self, filePath):
        # filename extension : xxxx.h5
        self.model = keras.models.load_model(filePath)

    # private Method-------------------------------------------------------

    # OrverrideTarget
    def InitializeModel(self, n):
        pass

    def PreProcessing(self, rawData, param):
        afterData = rawData
        return afterData
