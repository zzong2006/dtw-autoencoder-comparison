"""
       Test AutoEncoder Class
       정상적인 ECG 시계열 데이터를 학습하여 복원한다.
       데이터셋은 길이가 140인 총 4998 개의 시계열 데이터로, 0 (비정상) 또는 1 (정상)으로 labeling 되어 있다.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import ts_datset

from autoencoder import AutoEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

# Hyper-Parameter
USE_RNN = True
SHOW_DATASET = False
BATCH_SIZE = 512

dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
print(dataframe.head())

# 마지막 column 은 label 임
labels = raw_data[:, -1]

# 마지막 column을 제외한 나머지는 ecg data임
data = raw_data[:, 0: -1]

train_data, test_data, train_labels, test_labels = \
    train_test_split(data, labels, test_size=0.2, random_state=10)

# feature scaling
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
# test/validation data도 train_data와 동일한 scaling이 적용되야 함
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# label이 1로된 정상 ecg 데이터만 학습시킨다.
train_labels = (train_labels == 1)
test_labels = (test_labels == 1)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

if SHOW_DATASET:
    # 정상 ECG Plotting
    plt.grid()
    plt.plot(np.arange(140), normal_train_data[0])
    plt.title("A Normal ECG")
    plt.show()

    # 비정상 ECG Plotting
    plt.grid()
    plt.plot(np.arange(140), anomalous_train_data[0])
    plt.title("An Anomalous ECG")
    plt.show()

if not USE_RNN:
    # Dense 모델 구축
    _, input_size = train_data.shape
    ae_model = AutoEncoder([input_size, input_size // 2, input_size // 4, input_size // 8])
    ae_model.compile(optimizer='adam', loss='mae')

    # 모델 학습. validation은 training과 달리 전체(normal + abnormal) 테스트 set을 사용하여 평가
    history = ae_model.fit(normal_train_data, normal_train_data, epochs=20, batch_size=BATCH_SIZE,
                           validation_data=(test_data, test_data), shuffle=True)

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # 정상 ECG의 재구성 테스트
    decoded_imgs = ae_model(normal_test_data).numpy()

    plt.plot(normal_test_data[0], 'b')
    plt.plot(decoded_imgs[0], 'r')
    plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

if USE_RNN: # RNN 모델 (Seq2Seq) 구축
    _, input_size = train_data.shape

    # 데이터 셋 제작
    seq_train_data = tf.expand_dims(normal_train_data, -1)
    seq_test_data = tf.expand_dims(test_data, -1)

    train_dataset = tf.data.Dataset.from_tensor_slices(seq_train_data)
    train_dataset = train_dataset.batch(BATCH_SIZE)\
        .map(ts_datset.slide_one_step) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # 모델 생성
    ae_rnn_model = AutoEncoder([input_size], network_type='rnn')
    ae_rnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mse')

    # 학습
    history = ae_rnn_model.fit(train_dataset, epochs=10, validation_data=(seq_train_data, seq_train_data))

    plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # 정상 ECG의 재구성 테스트
    output_seqs = ae_rnn_model(seq_train_data)
    # remove last dimension
    output_seqs = np.reshape(output_seqs, newshape=[output_seqs.shape[0], -1])

    plt.plot(normal_test_data[0], 'b')
    plt.plot(output_seqs[0], 'r')
    plt.fill_between(np.arange(140), output_seqs[0], normal_test_data[0], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

