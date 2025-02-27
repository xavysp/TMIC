"""
By: xavysp
This architecture comes from TEED. It is used
for  image classification
"""
import keras
import tensorflow as tf

def k_smish2(x):
    inp = x
    xDot = keras.ops.tanh(keras.ops.log(1+keras.ops.sigmoid(x)))
    return inp * xDot

class k_smish(keras.layers.Layer):
    def call(self, inp):
        x= inp
        xDot = tf.math.tanh(tf.math.log(1+tf.sigmoid(x)))
        return inp * xDot

def data_augmentation(images):
    data_augmentation_layers = [
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomContrast(0.2)
    ]
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
def model_maker(input_shape, num_classes):
    input = keras.Input(shape=input_shape)# [None, 28,28,1]
    x = data_augmentation(input)
    # x = keras.layers.Rescaling(1. / 255)(x)
    # TMIC size
    #tmic_size->  48=small, 128 = medium, 256 = large
    m_size = 64 # model size from the third block
    f_size = 128 if m_size==48 else 128
    # block 1
    x = keras.layers.Conv2D(16,3, strides=2, padding="same")(input) # [None, 14,14,16]
    # x = keras.layers.Activation("smish")(x)
    x = k_smish()(x)
    x = keras.layers.Conv2D(16,3, strides=1, padding="same")(x)
    # x = keras.layers.Activation("relu")(x)
    x = k_smish()(x)
    #px = keras.layers.BatchNormalization(axis=-1)(x)
    ##xs1 =keras.layers.Conv2D(32,1, strides=2,padding="same")(x) #skep Connection # [None, 7,7,32]
    px = keras.layers.Dropout(0.25)(x)

    # Block 2
    # block3-1
    
    # flatten
    ex = keras.layers.Flatten()(px)
    # ex = keras.layers.Conv2D(32, 3, padding="same")(px)
    # ex = keras.layers.Activation("relu")(ex)
    # ex = keras.layers.GlobalAveragePooling2D()(ex)
    # We specify activation=None so it return logits
    ex = keras.layers.Dense(f_size,activation=k_smish2)(ex)
    ex= keras.layers.BatchNormalization()(ex)
    ex = keras.layers.Dropout(0.5)(ex)
    output = keras.layers.Dense(num_classes,activation="softmax")(ex)
    # output = keras.layers.Dense(units, activation=None)(x)
    return keras.Model(input, output)


