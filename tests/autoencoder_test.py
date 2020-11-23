import unittest
import numpy as np
import tensorflow as tf
import autoencoder
import matplotlib.pyplot as plt

from ts_datset import *


class autoencoder_test(unittest.TestCase):
    def test_rnn_type(self):
        # hyper-parameters
        hidden_size = 64
        epochs = 40
        batch_size = 8

        # dataset
        seq_length = 8  # 시퀀스 길이
        num_train = 32  # 학습 데이터 개수

        sample_data = tf.random.normal(shape=[num_train, seq_length, 1])
        dataset = tf.data.Dataset.from_tensor_slices(sample_data)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).map(slide_one_step)

        # model
        model = autoencoder.AutoEncoder([hidden_size], network_type='rnn')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.MeanSquaredError, metrics=["MSE"])
        # train model
        model.fit(dataset, epochs=epochs)
        # predict by model
        test_data = sample_data[:1]
        result = model(sample_data[:1])

        plt.plot(tf.reshape(test_data, shape=[-1]))
        plt.plot(tf.reshape(result, shape=[-1]))
        plt.show()


if __name__ == '__main__':
    unittest.main()
