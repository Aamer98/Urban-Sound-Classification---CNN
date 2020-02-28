import numpy as np
from numpy import genfromtxt

x_train = genfromtxt('/home/aamer/train_data.csv', delimiter=',')
y_train = genfromtxt('/home/aamer/train_labels.csv', delimiter=',')
x_test = genfromtxt('/home/aamer/test_data.csv', delimiter=',')
y_test = genfromtxt('/home/aamer/test_labels.csv', delimiter=',')
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
y_train.shape,y_test.shape
x_train=np.reshape(x_train,(x_train.shape[0], 40,5))
x_test=np.reshape(x_test,(x_test.shape[0], 40,5))
x_train.shape,x_test.shape

x_train=np.reshape(x_train,(x_train.shape[0], 40,5,1))
x_test=np.reshape(x_test,(x_test.shape[0], 40,5,1))

x_train.shape,x_test.shape

from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

model=Sequential()

model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(40,5,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(10,activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=50,epochs=30,validation_data=(x_test,y_test))
