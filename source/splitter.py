import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

if __name__ == '__main__':
    data = tf.random.normal(shape=[200, 100, 1])
    val_data = tf.random.normal(shape=[200, 100, 1])
    data = tf.cast(data, tf.float32)

    data_min = tf.reduce_min(data)
    data_max = tf.reduce_max(data)

    data = (data - data_min) / (data_max - data_min)
    val_data = (val_data - data_min) / (data_max - data_min)
    # %%
    tf.reduce_min(data)

    # %%
    temp_data = tf.concat([np.ones([200, 1, 1]) * 3, data], axis=1)
    val_temp_data = tf.concat([np.ones([200, 1, 1]) * 3, val_data], axis=1)
    shift_data = temp_data[:, :-1, :]
    shift_val_data = val_temp_data[:, :-1, :]
    data[0, :5], shift_data[0, :5]


    # %%
    dataset = tf.data.Dataset.from_tensor_slices(({"encoder_input": data, "decoder_input": shift_data}, data))
    dataset = dataset.shuffle(200).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # %%
    encoder_inputs = layers.Input(shape=(None, 1), name='encoder_input')

    encoder = layers.LSTM(50, return_state=True)
    _, h, c = encoder(encoder_inputs)

    encoder(data)[0].shape
    # %%
    decoder = layers.LSTM(50, return_sequences=True)
    decoder_inputs = layers.Input(shape=(None, 1))
    output = decoder(decoder_inputs, initial_state=[h, c])
    dense = layers.Dense(1, activation='sigmoid')
    output = dense(output)

    dense(decoder(shift_data)).shape

    # %%
    model = Model([encoder_inputs, decoder_inputs], output)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='mse', )
    model.add_metric(encoder_inputs,name="test")
    model.fit(dataset, validation_data=([val_data, shift_val_data], val_data), epochs=5, batch_size=20)
