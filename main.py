import tensorflow as tf
import keras

from model import model_maker

image_size = (28, 28,1)
batch_size = 32

model = model_maker(image_size,10)
keras.utils.plot_model(model, show_shapes=True)