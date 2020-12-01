import unittest
import numpy as np
import tensorflow as tf
import autoencoder
import matplotlib.pyplot as plt

from ts_datset import *


class AutoEncoderTest(unittest.TestCase):
    def test_rnn_type(self):
        # hyper-parameters
        hidden_size = 10
        epochs = 500
        batch_size = 8

        # dataset
        seq_length = 10  # 시퀀스 길이
        num_train = 32  # 학습 데이터 개수

        sample_data = tf.random.normal(shape=[num_train, seq_length, 1])
        sample_data2 = tf.convert_to_tensor(np.random.randint(0, 10, size=[10, 1]))
        sample_data2 = tf.reshape(sample_data2, shape=[1, -1, 1])

        dataset = tf.data.Dataset.from_tensor_slices(sample_data)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        # model
        model = autoencoder.AutoEncoder([hidden_size],
                                        network_type='rnn',
                                        time_steps=seq_length,
                                        num_of_features=1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.MeanSquaredError(), metrics=["MSE"])

        sample_output = model(sample_data)
        self.assertEqual(sample_output.shape, [num_train, seq_length, 1])

        # train model
        model.fit(x=sample_data2, y=sample_data2, epochs=epochs)

        # predict by model
        test_data = sample_data2[:1]
        result = model(sample_data2[:1])

        plt.plot(tf.reshape(test_data, shape=[-1]))
        plt.plot(tf.reshape(result, shape=[-1]))
        plt.show()


if __name__ == '__main__':
    unittest.main()
