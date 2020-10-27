import numpy as np
import tensorflow as tf


def training_loop(model, x, y, val_x, val_y, train_fn, loss_fn, lr, bs, es) -> list:
    """
    mini-batch gradient descent를 이용한 학습

    :param model: 학습할 모델 (오토인코더)
    :param x: 학습 데이터
    :param y: 타겟 데이터
    :param train_fn: 훈련에 사용할 함수
    :param loss_fn: loss function
    :param lr: 학습률
    :param bs: 배치 사이즈
    :param es: 학습 횟수 (epochs)
    :return: 일정 간격에 따른 평균 loss 값을 저장한 list (0: training, 1: validation)
    """
    ls = [[], []]
    loss_avg = []
    val_loss_avg = []
    x_m, _ = x.shape
    if x_m % bs == 0:
        total_step = x_m // bs
    else:
        total_step = x_m // bs

    for epoch in range(es):
        loss_avg.clear()
        val_loss_avg.clear()
        for j in range(total_step):
            if j != total_step - 1:
                sliced_x = x[j * bs: (j + 1) * bs]
                sliced_y = y[j * bs: (j + 1) * bs]
            else:
                sliced_x = x[j * bs:]
                sliced_y = y[j * bs:]
            train_fn(model, sliced_x, sliced_y, lr, loss_fn)
            loss_avg.append(loss_fn(sliced_y, model(sliced_x)))
        avg_loss = np.average(loss_avg)
        ls[0].append(avg_loss)
        print('{}) loss: {:.5f}'.format(epoch, avg_loss))
    return ls


def mse(t_y, p_y):
    return tf.reduce_mean(tf.square(t_y - p_y))
