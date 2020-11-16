import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

if __name__ == '__main__':
    data = np.arange(0, 100)
    data = tf.cast(data.reshape([len(data), 1]), tf.float32)

    data_min = tf.reduce_min(data)
    data_max = tf.reduce_max(data)

    data = (data - data_min) / (data_max - data_min)

    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data[:-10], data[10:], sequence_length=10, batch_size=32
    )
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(25))

    # for batch in dataset:
    #     inputs, targets = batch
    #     print(inputs.shape, model(inputs).shape)
    # batch_size, timesteps, feature
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss='mse', )
    model.fit(dataset, epochs=20)


