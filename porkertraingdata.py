import sys
from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

batch_size = 32
num_classes = 52
epochs = 100000

input_shape = (128, 128)

np.set_printoptions(threshold=sys.maxsize)

train_dir = "train/"
is_continue = True

datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=20,
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

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=input_shape,
    color_mode="grayscale",
    batch_size=batch_size,
    subset='validation'
)

if is_continue:
    model = load_model("TraingData.h5")
else:
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(*input_shape, 1)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

model.fit(train_generator,
          steps_per_epoch=8, epochs=epochs,
          validation_data=val_generator,
          validation_steps=1,
          callbacks=[tensorboard_callback])
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

model.save("TraingData.h5")
