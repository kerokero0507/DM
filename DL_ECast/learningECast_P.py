import pandas as pd
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import RMSprop, Adam
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/191217_dataset.csv',sep=',')
x = df.iloc[:, 1:42]
y = df.iloc[:, 44]

#print(x)
#print(y)

x_shifts = pd.get_dummies(x["item_1"], 4)
x = x.drop("item_1", axis=1)
x = x.drop("item_12", axis=1)

#print(x)
#print(x_shifts)


sc = preprocessing.StandardScaler()
sc.fit(x)
x_std = sc.transform(x)

a = x_shifts.as_matrix()
x_std = np.concatenate([x_std, a], 1)

#print(x_std)


x_tra, x_te, t_tra, t_te = train_test_split(x_std, y, test_size=0.2, random_state=0)
x_train = pd.DataFrame(x_tra)
x_test = pd.DataFrame(x_te)
t_train = pd.DataFrame(t_tra)
t_test = pd.DataFrame(t_te)

print(x_test)
print(t_test)


def reg_model():
	reg = Sequential()
	reg.add(Dense(40, input_dim=len(x_train.columns), activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.1)))
#	reg.add(Dropout(0.2))
	reg.add(Dense(20, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.1)))
	reg.add(Dense(1))
	
	reg.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-2), metrics=['accuracy'])
	reg.summary()

	return reg

#model = KerasRegressor(build_fn=reg_model, epochs=6, batch_size=127, verbose=0, validation=(x_test, t_test))
model = reg_model()

#model.fit(x_train, t_train)

epo = 2000
history = model.fit(x_train, t_train, batch_size=127, epochs=epo, verbose=1, validation_data=(x_test, t_test))
#model.score(x_test, t_test)
score = model.evaluate(x_test, t_test, verbose=0)
y_pred = model.predict(x_test)
print('test loss:', score[0])
print('test Accuracy:', score[1])
#y_pred = pd.DataFrame(y_pred)
#y_pred.to_csv('result.csv', index=False)
model.save('modeldata.h5')

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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(range(1, epo+1), acc, marker='.', label='accuracy')
plt.plot(range(1, epo+1), val_acc, marker='.', label='val_accuracy')

plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

###figure###
plt.figure()
plt.scatter(t_train, model.predict(x_train), label='train', c='blue')
plt.xlabel('trainData')
plt.ylabel('predictData')
plt.scatter(t_test, model.predict(x_test), label='test', c='lightgreen', alpha=0.8)
plt.legend(loc=4)
plt.xlim(250,450)
plt.ylim(250,450)
plt.title('rin-noudo')
plt.show()

######outputFile######
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('predict_test.csv', index=False)

x_pred = model.predict(x_train)
x_pred = pd.DataFrame(x_pred)
x_pred.to_csv('predict_train.csv', index=False)

t_train.to_csv('label_train.csv', index=False)
t_test.to_csv('label_test.csv', index=False)
