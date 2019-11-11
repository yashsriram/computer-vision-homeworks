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


available_indices_for_mini_batch = list(np.arange(12000))


def get_mini_batch(im_train, label_train, batch_size):
    global available_indices_for_mini_batch
    train_samples_size = im_train.shape[1]
    # If all are sampled restart sampling
    if len(available_indices_for_mini_batch) == 0:
        available_indices_for_mini_batch = list(np.arange(train_samples_size))

    mini_batch_x = []
    mini_batch_y = []
    np.random.shuffle(available_indices_for_mini_batch)
    for train_idx in available_indices_for_mini_batch:
        # Sample this index into mini batch
        mini_batch_x.append(im_train[:, train_idx])
        mini_batch_y.append(one_hot_encoding(label_train[0, train_idx]))
        # Stop after batch_size number of samples
        if len(mini_batch_x) == batch_size:
            break
    # Remove sampled indices from available_indices_for_mini_batch
    available_indices_for_mini_batch = available_indices_for_mini_batch[len(mini_batch_x):]

    mini_batch_x, mini_batch_y = np.transpose(np.asarray(mini_batch_x)), np.transpose(np.asarray(mini_batch_y))
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
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
    # TO DO
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
