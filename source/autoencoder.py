# keras로 만든 Deep autoEncoder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class AutoEncoder(Model):
    def __init__(self, num_of_hidden: list,
                 num_of_seqs: int = None,
                 network_type: str = 'normal',
                 num_of_features=1,
                 var_dim=None,
                 **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        self.type = network_type

        # Dense layer type Model
        if network_type == 'normal':
            # Encoder 모델 레이어 설정
            self.encoder.add(tf.keras.Input(shape=(num_of_hidden[0],)))  # 입력 array shape는 (None, units[0])
            for i in range(1, len(num_of_hidden)):
                self.encoder.add(layers.Dense(num_of_hidden[i], activation='relu'))

            # Decoder 모델 레이어 설정
            self.decoder.add(tf.keras.Input(shape=(num_of_hidden[-1],)))
            for i in reversed(range(1, len(num_of_hidden) - 1)):
                self.decoder.add(layers.Dense(num_of_hidden[i], activation='relu'))
            self.decoder.add(layers.Dense(num_of_hidden[0], activation='sigmoid'))

        elif network_type == 'rnn':  # LSTM AutoEncoder
            assert num_of_seqs is not None
            hidden_units = num_of_hidden[0]
            # Encoder 모델 레이어 설정
            self.encoder.add(tf.keras.Input(shape=(num_of_seqs, num_of_features)))
            self.encoder.add(layers.LSTM(hidden_units // 2, return_sequences=False, activation='relu'))
            self.encoder.add(layers.RepeatVector(num_of_seqs))

            # Decoder 모델 레이어 설정
            self.decoder.add(tf.keras.Input(shape=(num_of_seqs, hidden_units // 2)))
            self.decoder.add(layers.LSTM(hidden_units // 2, return_sequences=True, activation='relu'))
            self.decoder.add(layers.TimeDistributed(layers.Dense(num_of_features, activation='sigmoid')))

        elif network_type == 'cnn':  # convolutional autoencoder
            self.encoder.add(layers.Conv1D(num_of_hidden[0] // 8))
        elif network_type == 'var':  # variational autoencoder
            assert var_dim is not None
            # Encoder 모델 레이어 설정
            self.encoder.add(tf.keras.Input(shape=(num_of_hidden[0],)))  # 입력 array shape는 (None, units[0])
            for i in range(1, len(num_of_hidden)):
                self.encoder.add(layers.Dense(num_of_hidden[i], activation='relu'))
            self.z_mean = layers.Dense(var_dim)
            self.z_log_var = layers.Dense(var_dim)
            self.sampling = Sampling()

            # Decoder 모델 레이어 설정
            self.decoder.add(tf.keras.Input(shape=(var_dim,)))
            for i in reversed(range(1, len(num_of_hidden))):
                self.decoder.add(layers.Dense(num_of_hidden[i], activation='relu'))
            self.decoder.add(layers.Dense(num_of_hidden[0], activation='sigmoid'))

    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the reconstruction_loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred)
            if self.type == 'var':
                loss += sum(self.losses)    # Add KL Divergence regularization loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False, mask=None, **kwargs):
        decoded = None

        if self.type == 'normal':
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
        elif self.type == 'rnn':
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
        elif self.type == 'cnn':
            pass
        elif self.type == 'var':
            encoded = self.encoder(inputs)
            z_mean = self.z_mean(encoded)
            z_log_var = self.z_log_var(encoded)

            # KL divergence regularization loss 를 forward propagation에서 계산
            kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            self.add_loss(kl_loss)

            sampled = self.sampling((z_mean, z_log_var))
            decoded = self.decoder(sampled)
        return decoded

    def get_config(self):
        pass


class Sampling(tf.keras.layers.Layer):
    # A layer for Variational AutoEncoder
    # 평균과 분산을 이용하여 var_dim 차원의 vector를 sampling
    def __init__(self):
        super(Sampling, self).__init__()

    @tf.function
    def call(self, inputs, training=False, mask=None, **kwargs):
        z_mean = inputs[0]
        z_log_var = inputs[1]

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(shape=[batch, dim])  # noise
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class EncoderRNN(Model):
    def __init__(self, hidden):
        super(EncoderRNN, self).__init__()
        self.rnn = layers.LSTM(hidden, return_state=True)

    def get_config(self):
        pass

    def call(self, inputs, hidden=None, training=None, mask=None):
        if hidden is not None:
            output, state = self.rnn(inputs, initial_state=hidden)
        else:
            output, state = self.rnn(inputs)
        return output, state


class DecoderRNN(Model):
    def __init__(self, hidden):
        super(DecoderRNN, self).__init__()
        self.rnn = layers.LSTM(hidden, return_sequences=True, return_state=True)
        self.fc = layers.Dense(1)

    def get_config(self):
        pass

    def call(self, inputs, hidden):
        output, state = self.rnn(inputs, initial_state=hidden)
        output = self.fc(output)
        return output, state


if __name__ == '__main__':
    pass
