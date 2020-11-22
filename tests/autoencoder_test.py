import unittest
import numpy as np
import tensorflow as tf
import autoencoder

class autoencoder_test(unittest.TestCase):
    def test_rnn_training(self):
        # hyper-parameters
        hidden_size = 16
        epochs = 10

        # model
        model = autoencoder.AutoEncoder([hidden_size], network_type='rnn')

        # dataset
        seq_length = 50      # 시퀀스 길이
        num_train = 100      # 학습 데이터 개수
        np.random.random(size=(num_train, seq_length, 1))

        c = 10
        self.assertEqual(c, 10)



if __name__ == '__main__':
    unittest.main()