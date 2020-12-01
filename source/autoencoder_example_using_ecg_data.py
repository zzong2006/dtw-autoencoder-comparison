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
import math

from autoencoder import AutoEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

# Hyper-Parameter
NETWORK_TYPE = 'var'
SHOW_DATASET = False
BATCH_SIZE = 128

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

if NETWORK_TYPE == 'normal':
    # Sample 개수가 Batch size 개수랑 맞아 떨어지게 수정
    normal_train_data = normal_train_data[:(normal_train_data.shape[0] // BATCH_SIZE) * BATCH_SIZE]

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

    # loss: 0.0146 - val_loss: 0.0267
    plt.plot(normal_test_data[0], 'b')
    plt.plot(decoded_imgs[0], 'r')
    plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

if NETWORK_TYPE == 'rnn':  # RNN 모델 (Seq2Seq) 구축
    _, seq_length = train_data.shape    # ecg seq_legnth는 140
    epochs = 20
    features = 10                       # time step을 seq_length // features로 수정 (disjoint window 형식)
    assert seq_length % features == 0

    # 데이터 셋 제작 (학습 데이터 개수가 batch size의 배수여야 함)
    normal_train_data = normal_train_data[:(normal_train_data.shape[0] // BATCH_SIZE) * BATCH_SIZE]

    seq_train_data = tf.reshape(normal_train_data, [-1, seq_length // features, features])
    seq_test_data = tf.reshape(test_data, [-1, seq_length // features, features])

    # 모델 생성
    ae_rnn_model = AutoEncoder([seq_length], network_type='rnn', time_steps=seq_length // features, num_of_features=features)
    ae_rnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                         loss="mae")

    # 학습
    history = ae_rnn_model.fit(x=seq_train_data, y=seq_train_data, epochs=epochs, batch_size=BATCH_SIZE,
                               validation_data=(seq_test_data, seq_test_data), shuffle=True)

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # 정상 ECG의 재구성 테스트
    index = 10
    output_seqs = ae_rnn_model(seq_test_data)
    single_seq = output_seqs[index]
    # remove last dimension
    single_seq = np.reshape(single_seq, newshape=[-1])

    # Recorded loss: 0.0238 - val_loss: 0.0303
    plt.plot(normal_test_data[index], 'b')
    plt.plot(single_seq, 'r')
    plt.fill_between(np.arange(140), single_seq, normal_test_data[index], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

if NETWORK_TYPE == 'cnn':
    pass

if NETWORK_TYPE == 'var':
    # Variational(Generative) 모델 구축
    _, input_size = train_data.shape
    ae_model = AutoEncoder([input_size, input_size // 2, input_size // 4, input_size // 8],
                           network_type='var')
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

