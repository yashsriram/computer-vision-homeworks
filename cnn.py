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


sampled_indices_for_mini_batch = []


def get_mini_batch(im_train, label_train, batch_size):
    global sampled_indices_for_mini_batch
    num_train_samples = im_train.shape[1]
    # If all are sampled restart sampling
    if len(sampled_indices_for_mini_batch) == num_train_samples:
        sampled_indices_for_mini_batch = []

    mini_batch_x = []
    mini_batch_y = []
    shuffled_train_indices = np.arange(num_train_samples)
    for train_idx in shuffled_train_indices:
        # If this index already sampled then go with next one
        if train_idx in sampled_indices_for_mini_batch:
            continue
        # Sample this index into mini batch
        mini_batch_x.append(im_train[0, train_idx])
        mini_batch_y.append(one_hot_encoding(label_train[0, train_idx]))
        # Remember that this index is sampled
        sampled_indices_for_mini_batch.append(train_idx)
        # Stop after batch_size number of samples
        if len(mini_batch_x) == batch_size:
            break

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
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()
