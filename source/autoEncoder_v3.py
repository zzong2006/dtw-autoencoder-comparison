# keras로 만든 Deep autoEncoder

import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers

# 추후에 autoEncoder의 Layer를 조절할 수 있게 만들것임.
from sklearn import preprocessing


class nnAutoEncoder():
    def __init__(self, trainFile = 'P.csv', testFile = 'test_splitted_data.csv',
                 withLabel = False, reducedDim = 64):
        # Loads the training and test data sets (ignoring class labels)

        # window 데이터 준비
        df_train = pd.read_csv(trainFile, header=None)
        df_test = pd.read_csv(testFile, header = None)

        print(df_train.head())
        print(df_test.head())

        self.windowSize = df_train.shape[1]
        self.totalWindow = df_train.shape[0]

        if withLabel :
            self.true_labels_ = df_test.iloc[:, 0].values
            self.x_train = df_train.drop(df_train.columns[0], axis=1).values
            self.x_test = df_test.drop(df_test.columns[0], axis=1).values
            self.windowSize -= 1
        else :
            self.x_train = df_train.values
            self.x_test = df_test.values

        self.normalized_X_train = preprocessing.scale(self.x_train)
        self.normalized_X_test = preprocessing.scale(self.x_test)

        # (x_train.shape, x_test.shape)
        # ((200, 1000), (200, 1000))

        # input dimension = 1000
        input_dim = self.windowSize

        self.autoencoder = Sequential()

        # Encoder Layers
        self.autoencoder.add(Dense((int)(self.windowSize/2), input_shape=(input_dim,), activation='linear'))
        self.autoencoder.add(PReLU())
        self.autoencoder.add(Dense((int)(self.windowSize/4), activation='linear'))
        self.autoencoder.add(PReLU())
        self.autoencoder.add(Dense(reducedDim, activation='linear'))
        self.autoencoder.add(PReLU())

        # Decoder Layers
        self.autoencoder.add(Dense((int)(self.windowSize / 4), activation='linear'))
        self.autoencoder.add(PReLU())
        self.autoencoder.add(Dense((int)(self.windowSize/2), activation='linear'))
        self.autoencoder.add(PReLU())
        self.autoencoder.add(Dense(self.windowSize, activation='linear'))

        self.reducedDimension = self.autoencoder.get_layer('dense_3').output_shape[1]
        self.encoder = Model(inputs=self.autoencoder.input, outputs=self.autoencoder.get_layer('dense_3').output)

        adam = optimizers.adamax(lr=0.005, decay=0.0000001)
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        rmsprop = optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=None, decay=0.0)
        self.autoencoder.compile(optimizer=adam, metrics=['accuracy'], loss='mse')

    def train(self, epochs, batchSize):
        self.history = self.autoencoder.fit(self.normalized_X_train, self.normalized_X_train,
                             epochs=epochs,
                             batch_size=batchSize,
                             shuffle=True,
                             validation_data=(self.normalized_X_test, self.normalized_X_test))

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
        random_test_windows = np.random.randint(self.normalized_X_test.shape[0], size=num_windows)
        encoded_windows = self.encoder.predict(self.normalized_X_test)
        self.decoded_windows = self.autoencoder.predict(self.normalized_X_test)
        encoded_windows = np.reshape(encoded_windows, (-1, self.reducedDimension))
        diff_avg = []

        for i, window_idx in enumerate(random_test_windows):
            print('#'+str(i+1) + ': average of original & reconstructed data')
            print(str(np.mean(self.normalized_X_test[window_idx])) + ' & ' + str(
                np.mean(self.decoded_windows[window_idx] )))
            if withReducedData :
                print('Reduced data : ' + str(encoded_windows[i]))
            diff_avg.append(np.mean((self.normalized_X_test[window_idx])-(self.decoded_windows[window_idx])))
            print('Difference Average : "' + str(diff_avg[i]) + '"')

        print('Generally this encoder can make difference average : '+ str(np.mean(diff_avg)))
        print('Max difference : ' + str(np.max(np.abs(diff_avg))))
        print('Min difference : ' + str(np.min(np.abs(diff_avg))))

    def getReducedData(self):
        encoded_windows = self.encoder.predict(self.normalized_X_test)
        encoded_windows = np.reshape(encoded_windows, (-1, self.reducedDimension))
        return encoded_windows

    def showReducedData(self):
        encoded_windows = self.encoder.predict(self.normalized_X_test)
        encoded_windows = np.reshape(encoded_windows, (-1, self.reducedDimension))
        plt.figure(); plt.plot(self.normalized_X_test[4])
        plt.figure(); plt.plot(encoded_windows[4])


    def showResult(self):
        plot_loss(self.history)
        plt.show()
        plot_acc(self.history)
        plt.show()



def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()

def plot_loss(history, title=None):
    # summarize history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()

# autoencoder.summary()
#
# input_img = Input(shape=(input_dim,))
# encoder_layer1 = autoencoder.layers[0]
# encoder_layer2 = autoencoder.layers[1]
# encoder_layer3 = autoencoder.layers[2]
# encoder = Model(input_img, encoder_layer3(encoder_layer2(encoder_layer1(input_img))))

# encoder.summary()
#
# num_windows = 20
# random_test_windows = np.random.randint(x_test.shape[0], size=num_windows)
#
# encoded_windows = encoder.predict(x_test)
# decoded_windows = autoencoder.predict(x_test)
#
# sample_size = 10