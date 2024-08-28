"""
By: xavysp
This architecture comes from TEED. It is used
for  image classification
"""
import keras

import tensorflow as tf

import numpy as np


def model_maker(input_shape, num_classes):
    input = keras.Input(shape=input_shape)# [None, 28,28,1]
    # x = data_augmentation(input)
    # x = keras.layers.Rescaling(1. / 255)(x)
    # block 1
    x = keras.layers.Conv2D(16,3, strides=2, padding="same", activation="relu")(input) # [None, 14,14,16]
    x = keras.layers.Conv2D(16,3, strides=1, padding="same", activation="relu")(x)
    xs1 =keras.layers.Conv2D(32,1, strides=2,padding="same", activation="relu")(x) #skep Connection # [None, 7,7,32]

    # Block 2
    px = keras.layers.MaxPooling2D(3,2,"same")(x)
    px = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(px) # [None, 7,7,32]
    px = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(px)
    xs2 =keras.layers.Conv2D(48,1, strides=1, activation="relu")(px) #skep Connection 2

    # block3-1
    px = keras.layers.add([xs1,px])
    px = keras.layers.Conv2D(48, 3, padding="same", activation="relu")(px)
    px = keras.layers.Conv2D(48, 3, padding="same", activation="relu")(px)
    px = keras.layers.Average()([px,xs2])
    # block3-2
    px = keras.layers.Conv2D(48, 3, padding="same", activation="relu")(px)
    px = keras.layers.Conv2D(48, 3, padding="same", activation="relu")(px)
    px = keras.layers.Average()([px, xs2])


    # flatten
    ex = keras.layers.Flatten()(px)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes
    # x = keras.layers.Dropout(0.25)(ex)
    # We specify activation=None so it return logits
    ex = keras.layers.Dense(64,activation="relu")(ex)
    output = keras.layers.Dense(10,activation="softmax")(ex)
    # output = keras.layers.Dense(units, activation=None)(x)
    return keras.Model(input, output)


