# keras로 만든 Convolutional (1D) autoEncoder

import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers

# 추후에 autoEncoder의 Layer를 조절할 수 있게 만들것임.
class convAutoEncoder():
    def __init__(self, trainFile = 'splitted_data.csv', testFile = 'test_splitted_data.csv',
                 window_size = 1000, max_size = 255):
        # Loads the training and test data sets (ignoring class labels)
        self.windowSize = window_size
        self.maxSize = max_size

        # window 데이터 준비
        self.x_train = []
        self.x_test = []

        with open(trainFile, 'r') as reader:
            for i, line in enumerate(reader):
                self.x_train.append(line.split(','))

        with open(testFile, 'r') as reader:
            for i, line in enumerate(reader):
                self.x_test.append(line.split(','))

        # Normalization
        for line in self.x_train:
             line[:] = [float(x) for x in line]

        for line in self.x_test:
             line[:] = [float(x) for x in line]

        # you need to reshape you data to have a spatial dimension for Conv1d to make sense:
        self.x_train = np.reshape(self.x_train, (-1, window_size, 1))
        self.x_test = np.reshape(self.x_test, (-1, window_size, 1))

        self.x_train = self.x_train.astype('float32') / self.maxSize
        self.x_test = self.x_test.astype('float32') / self.maxSize

        self.total_window = self.x_test.shape[0]
        # (x_train.shape, x_test.shape)
        # ((200, 1000), (200, 1000))

        # input dimension = 1000
        input_dim = self.x_train.shape[1]
        encoding_dim = 32

        self.autoencoder = Sequential()

        # Encoder Layers
        self.autoencoder.add(Conv1D(encoding_dim , 10, activation='linear',
                                padding='same', input_shape=(window_size,1)))
        self.autoencoder.add(LeakyReLU(alpha=0.001))
        self.autoencoder.add(MaxPooling1D(5, padding='same'))
        self.autoencoder.add(Conv1D(encoding_dim , 10, activation='linear',
                               padding='same'))
        self.autoencoder.add(LeakyReLU(alpha=0.001))
        self.autoencoder.add(MaxPooling1D(10, padding='same'))
        self.autoencoder.add(Conv1D(encoding_dim * 2, 10, activation='linear', padding='same'))
        self.autoencoder.add(LeakyReLU(alpha=0.001))
        self.autoencoder.add(MaxPooling1D(10, padding='same'))
        self.autoencoder.add(Conv1D(1, 10, activation='linear', padding='same'))
        self.autoencoder.add(LeakyReLU(alpha=0.001))

        # Decoder Layers
        self.autoencoder.add(UpSampling1D(10))
        self.autoencoder.add(Conv1D(encoding_dim * 2, 10, activation='linear',
                                padding='same'))
        self.autoencoder.add(LeakyReLU(alpha=0.001))
        self.autoencoder.add(UpSampling1D(10))
        self.autoencoder.add(Conv1D(encoding_dim , 10, activation='linear',
                                padding='same'))
        self.autoencoder.add(LeakyReLU(alpha=0.001))
        self.autoencoder.add(UpSampling1D(5))
        self.autoencoder.add(Conv1D(1, 10, activation='linear',padding='same'))
        self.autoencoder.add(LeakyReLU(alpha=0.001))

        self.reducedDimension = self.autoencoder.get_layer('conv1d_4').output_shape[1]

        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('conv1d_4').output)
        self.autoencoder.compile(optimizer='Adamax', loss='mean_squared_error')

    def train(self):
        self.autoencoder.fit(self.x_train, self.x_train,
                             epochs=100,
                             batch_size=64,
                             validation_data=(self.x_test, self.x_test))

        print("Training history")
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        plt.plot(self.autoencoder.history.history['loss'])
        ax1.set_title('loss')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(self.autoencoder.history.history['val_loss'])
        ax2.set_title('validation loss')

        plt.show()

    def showSummary(self, withEncoder = 1):
        self.autoencoder.summary()
        if withEncoder :
            self.encoder.summary()

    def showComprassPrecision(self, sample_size = 200, withReducedData = 0):
        # If you want check more samples, please increase the number of window samples
        num_windows = sample_size
        np.random.seed()
        random_test_windows = np.random.randint(self.x_test.shape[0], size=num_windows)
        encoded_windows = self.encoder.predict(self.x_test)
        decoded_windows = self.autoencoder.predict(self.x_test)
        encoded_windows = np.reshape(encoded_windows, (-1, self.reducedDimension))
        diff_avg = []

        for i, window_idx in enumerate(random_test_windows):
            print('#'+str(i+1) + ': average of original & reconstructed data')
            print(str(np.mean(self.x_test[window_idx] * self.maxSize)) + ' & ' + str(np.mean(decoded_windows[window_idx] * self.maxSize)))
            if withReducedData :
                print('Reduced data : ' + str(encoded_windows[i]))
            diff_avg.append(np.mean((self.x_test[window_idx]*self.maxSize)-(decoded_windows[window_idx]*self.maxSize)))
            print('Difference Average : "' + str(diff_avg[i]) + '"')


        print('Generally this encoder can make difference average : '+ str(np.mean(diff_avg)))
        print('Max difference : ' + str(np.max(np.abs(diff_avg))))
        print('Min difference : ' + str(np.min(np.abs(diff_avg))))

    def getReducedData(self):
        encoded_windows = self.encoder.predict(self.x_test)
        encoded_windows = np.reshape(encoded_windows, (-1, self.reducedDimension))
        return encoded_windows
