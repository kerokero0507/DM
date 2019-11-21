import pandas as pd
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import RMSprop, Adam
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('191101_testdata.csv',sep=',')
x = df.iloc[:, 0:10]
y = df.iloc[:, 10:11]

sc = preprocessing.StandardScaler()
sc.fit(x)
x_std = sc.transform(x)

x_tra, x_te, t_tra, t_te = train_test_split(x_std, y, test_size=0.2, random_state=0)
x_train = pd.DataFrame(x_tra)
x_test = pd.DataFrame(x_te)
t_train = pd.DataFrame(t_tra)
t_test = pd.DataFrame(t_te)

#print(x_test)
#print(t_test)
#print(x_test.columns)


#x_train = pd.read_csv('xtrain.csv', sep=',')
#t_train = pd.read_csv('ttrain.csv', sep=',')
#x_test = pd.read_csv('xtest.csv', sep=',')
#t_test = pd.read_csv('ttest.csv', sep=',')

#x_train = x_train.join(pd.get_dummies(x_train["item0"], 4))
#x_test = x_test.join(pd.get_dummies(x_test["item0"], 4))

#x_train = x_train.drop("item0", axis=1)
#x_test = x_test.drop("item0", axis=1)


def reg_model():
	reg = Sequential()
	reg.add(Dense(100, input_dim=len(x_train.columns), activation='relu'))

	reg.add(Dense(50, activation='relu'))
	reg.add(Dense(50, activation='relu'))
	reg.add(Dense(50, activation='relu'))
	reg.add(Dense(50, activation='relu'))
	reg.add(Dense(50, activation='relu'))
	reg.add(Dense(50, activation='relu'))
	reg.add(Dense(50, activation='relu'))
#	reg.add(Dropout(0.2))
	reg.add(Dense(1, activation='relu'))
	
	reg.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-2), metrics=['mae'])
#	reg.summary()

	return reg

#model = KerasRegressor(build_fn=reg_model, epochs=6, batch_size=127, verbose=0, validation=(x_test, t_test))
model = reg_model()

#model.fit(x_train, t_train)

epo = 100
history = model.fit(x_train, t_train, batch_size=1531, epochs=epo, verbose=1, validation_data=(x_test, t_test))
#model.score(x_test, t_test)
score = model.evaluate(x_test, t_test, verbose=0)
y_pred = model.predict(x_test)
print('test loss:', score[0])
print('test mae:', score[1])
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('result.csv', index=False)
model.save('modeldata.h5')

############kunren-data###############
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('result_kunren.csv', index=False)

############show Figure###############
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(1, epo+1), loss, marker='.', label='loss')
plt.plot(range(1, epo+1), val_loss, marker='.', label='val_loss')

plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

acc = history.history['mae']
val_acc = history.history['val_mae']

plt.plot(range(1, epo+1), acc, marker='.', label='mae')
plt.plot(range(1, epo+1), val_acc, marker='.', label='val_mae')

plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('mae')
plt.show()

###figure###
plt.figure()
plt.scatter(t_train, model.predict(x_train), label='train', c='blue')
plt.xlabel('trainData')
plt.ylabel('predictData')
plt.scatter(t_test, model.predict(x_test), label='test', c='lightgreen', alpha=0.8)
plt.legend(loc=4)
plt.xlim(400,800)
plt.ylim(400,800)
plt.title('kenmaryou')
plt.show()


