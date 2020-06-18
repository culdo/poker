import sys
from datetime import datetime

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_grads=True)

batch_size = 64
num_classes = 13
epochs = 100
# input image dimensions
img_rows, img_cols = 128, 128

input_shape = (img_rows, img_cols)

np.set_printoptions(threshold=sys.maxsize)
# data = np.ones((26, 3, img_rows, img_cols))
x_train = np.load('data.npy')
# test = np.ones((13, 3, img_rows, img_cols))
x_test = np.load('test.npy')
for i, x in enumerate(x_train):
    x_train[i] = x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114
x_train = np.array(x_train)
for i, x in enumerate(x_test):
    x_test[i] = x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114
x_train = np.array(x_train)
x_test = np.array(x_test)
# print(x_train[0][0],x_train[0][0].shape)
# print(x_train[2].shape)
# print(x_train[0])
# plt.imshow(x_train[1999])


x = int(len(data) / 13)

y_train = np.zeros(int(len(data) / 13))
# y_train = np.append(y_train,np.ones(int(len(data)/13)))
for i in range(12):
    p = i + 1
    Tdata = np.full(x, p, )
    y_train = np.append(y_train, Tdata)

# print(y_train)

y = int(len(test) / 13)
y_test = np.zeros(int(len(test) / 13))
# y_test = np.append(y_test,np.ones(int(len(test)/13)))
for i in range(12):
    p = i + 1
    Ttest = np.full(y, p)
    y_test = np.append(y_test, Ttest)

# print(y_test)

x_train = x_train.reshape((x_train.shape[0], img_rows, img_cols, 1))
x_test = x_test.reshape((x_test.shape[0], img_rows, img_cols, 1))
for img in x_test:
    cv2.imshow("img", img)
    cv2.waitKey()
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# # convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(16, (1, 1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(32, (1, 1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.add(Conv2D(32, kernel_size=(3, 3),
#                   activation='relu',
#                   input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("TraingData.h5")
