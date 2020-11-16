from autoencoder import AutoEncoderModule
from utils import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist_dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
m, rows, cols = x_train.shape

# Flatten
x_train = tf.cast(tf.reshape(x_train, [x_train.shape[0], -1]), tf.float32)
x_test = tf.cast(tf.reshape(x_test, [x_test.shape[0], -1]), tf.float32)

# Feature scaling (min-max)
max_val = np.amax(x_train, axis=1, keepdims=True)
min_val = np.amin(x_train, axis=1, keepdims=True)
x_train = (x_train - min_val) / (max_val - min_val)

max_val = np.amax(x_test, axis=1, keepdims=True)
min_val = np.amin(x_test, axis=1, keepdims=True)
x_test = (x_test - min_val) / (max_val - min_val)

input_size = cols * rows
layer_size_list = [input_size, input_size // 2, input_size // 4]
learning_rate = 0.01
epochs = 50
batch_size = 256   # going to use Mini-batch GD

deep_ae = AutoEncoderModule(layer_size_list)
loss_output = training_loop(deep_ae, x_train, x_train, deep_ae.train,
                            mse, learning_rate, batch_size, epochs)
plt.plot(range(epochs), loss_output, "r")
plt.show()

fig = plt.figure(figsize=(6, 2))
n = 2
