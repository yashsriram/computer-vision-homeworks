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
    dl_dw = dl_dw.reshape(-1)
    # dy_db = I (n x n)
    dl_db = dl_dy
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l = np.sum((y_tilde - y) ** 2)
    dl_dy = (y_tilde - y) * 2
    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp)
    y_tilde = np.asarray([(x_exp_i / x_exp_sum) for x_exp_i in x_exp])
    # Cross entropy loss Sig(y[i] * log(y_tilde[i]))
    l = 0
    for i, yi in enumerate(y):
        if yi > 0:
            l += yi * np.log(y_tilde[i])
    # Directly given in slides
    dl_dx = y_tilde - y
    return l, dl_dx


def relu(x):
    x_vectorized = x.reshape(-1)
    y = np.asarray([max(0, xi) for xi in x_vectorized])
    y = y.reshape(x.shape)
    return y


def relu_backward(dl_dy, x, y):
    dl_dx = np.zeros(dl_dy.shape)
    for i, dl_dy_i in enumerate(dl_dy):
        if dl_dy_i >= 0:
            dl_dx[i] = dl_dy_i
        else:
            dl_dx[i] = 0
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


def train_slp_linear(mini_batches_x, mini_batches_y):
    LEARNING_RATE = 0.001
    DECAY_RATE = 0.9
    NUM_ITERATIONS = 2000
    OUTPUT_SIZE = 10
    INPUT_SIZE = 196
    w = np.random.normal(0, 1, size=(OUTPUT_SIZE, INPUT_SIZE))
    b = np.random.normal(0, 1, size=(OUTPUT_SIZE, 1))
    num_mini_batches = len(mini_batches_x)
    loss_values = []
    for iter_i in range(NUM_ITERATIONS):
        print('Iteration # {}/{}\r'.format(iter_i, NUM_ITERATIONS), end='')
        if iter_i + 1 % 1000 == 0:
            LEARNING_RATE = LEARNING_RATE * DECAY_RATE
        # Determining current mini-batch
        curr_mini_batch_x = mini_batches_x[iter_i % num_mini_batches]
        curr_mini_batch_y = mini_batches_y[iter_i % num_mini_batches]
        curr_mini_batch_size = curr_mini_batch_x.shape[1]
        # Current mini-batch gradients
        mini_batch_dl_dw = np.zeros(OUTPUT_SIZE * INPUT_SIZE)
        mini_batch_dl_db = np.zeros(OUTPUT_SIZE)
        mini_batch_loss = 0
        for idx in range(curr_mini_batch_size):
            x = curr_mini_batch_x[:, idx]
            y = curr_mini_batch_y[:, idx]
            # Forward prop
            y_tilde = fc(x.reshape(INPUT_SIZE, 1), w, b).reshape(-1)
            # Loss
            l, dl_dy = loss_euclidean(y_tilde, y)
            mini_batch_loss += np.abs(l)
            # Back prop
            _, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            # Sum gradients
            mini_batch_dl_dw += dl_dw
            mini_batch_dl_db += dl_db
        # Update weights
        loss_values.append(mini_batch_loss)
        w = w - mini_batch_dl_dw.reshape(w.shape) * LEARNING_RATE
        b = b - mini_batch_dl_db.reshape(b.shape) * LEARNING_RATE

    print()
    plt.xlabel('iterations', fontsize=18)
    plt.ylabel('training loss', fontsize=16)
    plt.plot(loss_values)
    plt.show()

    return w, b


def train_slp(mini_batches_x, mini_batches_y):
    LEARNING_RATE = 0.04
    DECAY_RATE = 0.9
    NUM_ITERATIONS = 2000
    OUTPUT_SIZE = 10
    INPUT_SIZE = 196
    w = np.random.normal(0, 1, size=(OUTPUT_SIZE, INPUT_SIZE))
    b = np.random.normal(0, 1, size=(OUTPUT_SIZE, 1))
    num_mini_batches = len(mini_batches_x)
    loss_values = []
    for iter_i in range(NUM_ITERATIONS):
        print('Iteration # {}/{}\r'.format(iter_i, NUM_ITERATIONS), end='')
        if iter_i + 1 % 1000 == 0:
            LEARNING_RATE = LEARNING_RATE * DECAY_RATE
        # Determining current mini-batch
        curr_mini_batch_x = mini_batches_x[iter_i % num_mini_batches]
        curr_mini_batch_y = mini_batches_y[iter_i % num_mini_batches]
        curr_mini_batch_size = curr_mini_batch_x.shape[1]
        # Current mini-batch gradients
        mini_batch_dl_dw = np.zeros(OUTPUT_SIZE * INPUT_SIZE)
        mini_batch_dl_db = np.zeros(OUTPUT_SIZE)
        mini_batch_loss = 0
        for idx in range(curr_mini_batch_size):
            x = curr_mini_batch_x[:, idx]
            y = curr_mini_batch_y[:, idx]
            # Forward prop
            y_tilde = fc(x.reshape(INPUT_SIZE, 1), w, b).reshape(-1)
            # Loss
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            mini_batch_loss += np.abs(l)
            # Back prop
            _, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            # Sum gradients
            mini_batch_dl_dw += dl_dw
            mini_batch_dl_db += dl_db
        # Update weights
        loss_values.append(mini_batch_loss)
        w = w - mini_batch_dl_dw.reshape(w.shape) * LEARNING_RATE
        b = b - mini_batch_dl_db.reshape(b.shape) * LEARNING_RATE

    print()
    plt.xlabel('iterations', fontsize=18)
    plt.ylabel('training loss', fontsize=16)
    plt.plot(loss_values)
    plt.show()

    return w, b


def train_mlp(mini_batches_x, mini_batches_y):
    LEARNING_RATE = 0.01
    DECAY_RATE = 0.9
    NUM_ITERATIONS = 200
    OUTPUT_SIZE = 10
    MIDDLE_LAYER_SIZE = 30
    INPUT_SIZE = 196
    w1 = np.random.normal(0, 1, size=(MIDDLE_LAYER_SIZE, INPUT_SIZE))
    b1 = np.random.normal(0, 1, size=(MIDDLE_LAYER_SIZE, 1))
    w2 = np.random.normal(0, 1, size=(OUTPUT_SIZE, MIDDLE_LAYER_SIZE))
    b2 = np.random.normal(0, 1, size=(OUTPUT_SIZE, 1))
    num_mini_batches = len(mini_batches_x)
    loss_values = []
    for iter_i in range(NUM_ITERATIONS):
        print('Iteration # {}/{}\r'.format(iter_i, NUM_ITERATIONS), end='')
        if iter_i + 1 % 1000 == 0:
            LEARNING_RATE = LEARNING_RATE * DECAY_RATE
        # Determining current mini-batch
        curr_mini_batch_x = mini_batches_x[iter_i % num_mini_batches]
        curr_mini_batch_y = mini_batches_y[iter_i % num_mini_batches]
        curr_mini_batch_size = curr_mini_batch_x.shape[1]
        # Current mini-batch gradients
        mini_batch_dl_dw1 = np.zeros(MIDDLE_LAYER_SIZE * INPUT_SIZE)
        mini_batch_dl_db1 = np.zeros(MIDDLE_LAYER_SIZE)
        mini_batch_dl_dw2 = np.zeros(OUTPUT_SIZE * MIDDLE_LAYER_SIZE)
        mini_batch_dl_db2 = np.zeros(OUTPUT_SIZE)
        mini_batch_loss = 0
        for idx in range(curr_mini_batch_size):
            x = curr_mini_batch_x[:, idx]
            y = curr_mini_batch_y[:, idx]
            # FC 1
            fc_x = fc(x.reshape(INPUT_SIZE, 1), w1, b1).reshape(-1)
            # ReLU 1
            relu_fc_x = relu(fc_x)
            # FC 2
            fc_relu_fc_x = fc(relu_fc_x.reshape(MIDDLE_LAYER_SIZE, 1), w2, b2).reshape(-1)
            # ReLU 2
            relu_fc_relu_fc_x = relu(fc_relu_fc_x)
            # Loss
            l, dl_dy = loss_cross_entropy_softmax(relu_fc_relu_fc_x, y)
            mini_batch_loss += np.abs(l)
            # Back prop
            # ReLu 2
            dl_dy = relu_backward(dl_dy, fc_relu_fc_x, relu_fc_relu_fc_x)
            # FC 2
            dl_dy, dl_dw2, dl_db2 = fc_backward(dl_dy, relu_fc_x, w2, b2, fc_relu_fc_x)
            # ReLu 1
            dl_dy = relu_backward(dl_dy, fc_x, relu_fc_x)
            # FC 2
            _, dl_dw1, dl_db1 = fc_backward(dl_dy, x, w1, b1, fc_x)
            # Sum gradients
            mini_batch_dl_dw1 += dl_dw1
            mini_batch_dl_db1 += dl_db1
            mini_batch_dl_dw2 += dl_dw2
            mini_batch_dl_db2 += dl_db2
        loss_values.append(mini_batch_loss)
        # Update
        w1 = w1 - mini_batch_dl_dw1.reshape(w1.shape) * LEARNING_RATE
        b1 = b1 - mini_batch_dl_db1.reshape(b1.shape) * LEARNING_RATE
        w2 = w2 - mini_batch_dl_dw2.reshape(w2.shape) * LEARNING_RATE
        b2 = b2 - mini_batch_dl_db2.reshape(b2.shape) * LEARNING_RATE

    print()
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.xlabel('iterations', fontsize=18)
    plt.ylabel('training loss', fontsize=16)
    plt.plot(loss_values)
    plt.show()

    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    np.random.seed(42)
    # main.main_slp_linear()
    # main.main_slp()
    main.main_mlp()
    # main.main_cnn()
