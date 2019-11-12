import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def one_hot_encoding(number):
    """
    :param number: label 0 - 9
    :return:
    """
    assert 0 <= number <= 9
    encoding = [0] * 10
    encoding[number] = 1
    return encoding


def get_mini_batch(im_train, label_train, batch_size):
    train_samples_size = im_train.shape[1]
    shuffed_indices = list(np.arange(train_samples_size))
    np.random.shuffle(shuffed_indices)

    mini_batches_x = []
    mini_batches_y = []
    mini_batch_x = []
    mini_batch_y = []
    for train_idx in shuffed_indices:
        # Sample this index into mini batch
        mini_batch_x.append(im_train[:, train_idx])
        mini_batch_y.append(one_hot_encoding(label_train[0, train_idx]))
        # Stop after batch_size number of samples
        if len(mini_batch_x) == batch_size:
            mini_batch_x, mini_batch_y = np.transpose(np.asarray(mini_batch_x)), np.transpose(np.asarray(mini_batch_y))
            mini_batches_x.append(mini_batch_x)
            mini_batches_y.append(mini_batch_y)
            mini_batch_x = []
            mini_batch_y = []
    if len(mini_batch_x) > 0:
        mini_batch_x, mini_batch_y = np.transpose(np.asarray(mini_batch_x)), np.transpose(np.asarray(mini_batch_y))
        mini_batches_x.append(mini_batch_x)
        mini_batches_y.append(mini_batch_y)

    return mini_batches_x, mini_batches_y


def fc(x, w, b):
    y = np.matmul(w, x) + b
    return y


def fc_backward(dl_dy, x, w, b, y):
    n, m = w.shape
    # dy_dx = w
    dl_dx = np.matmul(dl_dy, w)
    # dy_dw =
    # [ X 0 0 0 ... 0 (n times)]
    # [ 0 X 0 0 ... 0 (n times)]
    #           (n times)
    # [ 0 0 0 0 ... X (n times)]
    dl_dw = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dl_dw[i, j] = dl_dy[i] * x[j]
    dl_dy.reshape(-1)
    # dy_db = I (n x n)
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l = np.sum((y_tilde - y) ** 2)
    dl_dy = (y_tilde - y) * 2
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    # TO DO
    return l, dl_dy


def relu(x):
    # TO DO
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    return dl_dw, dl_db


def pool2x2(x):
    # TO DO
    return y


def pool2x2_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def flattening(x):
    # TO DO
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    LEARNING_RATE = 0.1
    DECAY_RATE = 0.1
    return w, b


def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    return w, b


def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    np.random.seed(42)
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()
