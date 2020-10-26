#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

num_classes = 52
batch_size = num_classes
epochs = 100000

input_shape = (128, 128)

np.set_printoptions(threshold=sys.maxsize)

train_dir = "train/"
test_dir = "test/"
is_continue = False

datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=input_shape,
    color_mode="grayscale",
    batch_size=batch_size,
    subset='training'
)
print(train_generator.class_indices)
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=input_shape,
    color_mode="grayscale",
    batch_size=batch_size,
    subset='validation'
)
print(val_generator.class_indices)


def scheduler(epoch, lr):
    # if epoch < 200:
    #     return lr
    # else:
    #     lr = lr * tf.math.exp(-0.001)
    #     return lr
    return lr


cb_lr_sch = tf.keras.callbacks.LearningRateScheduler(scheduler)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="poker_model.h5",
    verbose=1,
    save_best_only=True,
    save_freq="epoch")
cb_early_stop = tf.keras.callbacks.EarlyStopping(patience=500, verbose=1)

if is_continue:
    model = load_model("poker_model.h5")
else:
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(*input_shape, 1)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    input_tensor = keras.Input((*input_shape, 1))
    x = Conv2D(64, (3, 3), activation="elu")(input_tensor)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (1, 1), activation="elu")(x)
    x = Conv2D(64, (3, 3), activation="elu")(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Conv2D(32, (1, 1), activation="elu")(x)
    x = Conv2D(128, (3, 3), activation="elu")(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (1, 1), activation="elu")(x)
    x = Conv2D(128, (3, 3), activation="elu")(x)
    x = Conv2D(64, (1, 1), activation="elu")(x)
    x = Conv2D(128, (3, 3), activation="elu")(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Flatten()(x)
    x = Dense(128, activation="elu")(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
print(model.summary())
hist = model.fit(train_generator,
                 steps_per_epoch=train_generator.n / batch_size, epochs=epochs,
                 validation_data=val_generator,
                 validation_steps=val_generator.n / batch_size,
                 callbacks=[model_checkpoint_callback, cb_early_stop, tensorboard_callback, cb_lr_sch])
for key in hist.history:
    print(key)
print("min_val_loss: %.4f" % min(hist.history["val_loss"]))
print("max_val_accuracy: %.4f" % max(hist.history["val_acc"]))
subprocess.call("notify-send -i %s -t %d %s '%s'"
                % ("/home/lab-pc1/nptu/lab/computer_vision/service/darknet_notext.png",
                   3000, "Poker", "min_val_loss: %.4f, max_val_accuracy: %.4f" % (
                   min(hist.history["val_loss"]), max(hist.history["val_accuracy"]))), shell=True)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
