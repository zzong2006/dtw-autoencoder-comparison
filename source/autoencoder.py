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
    def __init__(self, num_of_hidden: list, network_type: str = 'normal', **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        self.type = network_type
        self.loss_tracker = tf.keras.metrics.MeanSquaredError(name="loss")
        self.val_loss_tracker = tf.keras.metrics.MeanSquaredError(name="val_loss")

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
        # RNN Layer type Model (Seq2Seq)
        elif network_type == 'rnn':
            hidden_units = num_of_hidden[0] // 4
            self.encoder = EncoderRNN(hidden_units)
            self.decoder = DecoderRNN(hidden_units)
        elif network_type == 'cnn':
            self.encoder.add(layers.Conv1D(num_of_hidden[0] // 8))

    def call(self, inputs, training=False, mask=None, **kwargs):
        decoded = None

        if self.type == 'normal':
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
        elif self.type == 'rnn':
            if training:
                encoder_input, decoder_input = inputs[0], inputs[1]
                encoded, state_h, state_c = self.encoder(encoder_input)
                decoded = self.decoder(decoder_input, hidden=[state_h, state_c])
            else:
                encoder_input = inputs
                batch, time_step, features = encoder_input.shape
                encoded, state_h, state_c = self.encoder.predict(encoder_input)
                decoder_input = tf.zeros_like(encoder_input)
                decoder_input = decoder_input[:, :1, :]
                decoded = None
                for i in range(time_step):
                    decoder_input, state_h, state_c = self.decoder.predict(decoder_input, hidden=[state_h, state_c])

                    if decoded is not None:
                        decoded = tf.concat([decoded, decoder_input], axis=1)
                    else:
                        decoded = tf.identity(decoder_input)

        return decoded

    def get_config(self):
        pass

    @tf.function
    def train_step(self, tr_data):
        inp, target = tr_data

        with tf.GradientTape() as tape:
            predictions = self(inp, training=True)
            loss = tf.keras.losses.mean_squared_error(target, predictions)

        # Compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Compute own metrics
        self.loss_tracker.update_state(target, predictions)

        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]


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
