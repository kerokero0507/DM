import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split

from NNSubModule import BeforeProcess

class NeuralNetwork:
    def __init__(self, epocs, batchSize, learnigRate, modelName):
        self.epoc = epocs
        self.size = batchSize
        self.rate = learnigRate
        self.name = modelName

        self.model = Sequential()

    def Run(self, rawData, labelData, param, testSize):
        #beforeProcess
        treatmentData = self.PreProcessing(rawData, param)
        #-------------
        histryResut, scoreResut = self.Learning(treatmentData, labelData, testSize)

        return histryResut, scoreResut

    def Learning(self, teachData, labelData, testSize):
        self.InitializeModel()

        x_train, x_test, t_train, t_test = train_test_split(teachData, labelData, test_size=self.testSize, random_state=0)

        history = self.model.fit(x_train, t_train, batch_size=self.size, epochs=self.epoc, verbose=1, validation_data=(x_test, t_test))
        score = self.model.evaluate(x_test, t_test, verbose=0)
        return history, score

    def Predict(self, inputData):
        result = self.model.predict(inputData)
        return result

    def Save(self, filePath):
        #filename extension : xxxx.h5
        self.model.save(filePath)

    def Load(self, filePath):
        #filename extension : xxxx.h5
        self.model = keras.models.load_model(filePath)

    #private Method-------------------------------------------------------

    #OrverrideTarget
    def InitializeModel(self):
        pass

    def PreProcessing(self, rawData, param):
        afterData = rawData
        return afterData
    
    



    
    
