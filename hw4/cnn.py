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
    dl_dw = np.zeros(w.shape)
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


RELU_E = 0


def relu(x):
    x_vectorized = x.reshape(-1)
    y = np.zeros(x_vectorized.shape)
    for i, xi in enumerate(x_vectorized):
        if xi >= 0:
            y[i] = xi
        else:
            y[i] = xi * RELU_E
    y = y.reshape(x.shape)
    return y


def relu_backward(dl_dy, x, y):
    dl_dy_flat = dl_dy.reshape(-1)
    dl_dx = np.zeros(dl_dy_flat.shape)
    for i, dl_dy_i in enumerate(dl_dy_flat):
        if dl_dy_i >= 0:
            dl_dx[i] = dl_dy_i
        else:
            dl_dx[i] = dl_dy_i * RELU_E
    dl_dx = dl_dx.reshape(dl_dy.shape)
    return dl_dx


def conv(x, w_conv, b_conv):
    h, w, c1, c2 = w_conv.shape
    # assert that filters have odd numbered dimensions so that there is a single center point
    assert (h % 2 == 1)
    assert (w % 2 == 1)
    # pad the H and W axis not the C axis with appropriate number of zeros as determined by h and w axes of filter
    padding_h = int(h / 2)
    padding_w = int(w / 2)
    padded_x = np.pad(x, ((padding_h, padding_h), (padding_w, padding_w), (0, 0)), mode='constant')

    H_padded, W_padded, _ = padded_x.shape
    H, W, _ = x.shape
    y = np.zeros((H, W, c2))
    for c2_i in range(c2):
        filter_i = w_conv[:, :, :, c2_i]
        # iterating in padded input with pivot as original input cells
        for pos_h in range(padding_h, H_padded - padding_h):
            for pos_w in range(padding_w, W_padded - padding_w):
                # determine voxel from input to be convolved with the filter
                # note: padding_h is also equal to step(h / 2)
                # note: padding_w is also equal to step(w / 2)
                voxel = padded_x[pos_h - padding_h: pos_h + padding_h + 1,
                        pos_w - padding_w: pos_w + padding_w + 1, :]
                # convolve (element wise multiply and sum) filter and voxel
                conved = np.sum(np.multiply(voxel, filter_i))
                # add bias
                conved += b_conv[c2_i]
                # put the convolved value in proper position in output
                y[pos_h - padding_h, pos_w - padding_w, c2_i] = conved
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # assert dimension sanity
    assert (dl_dy.shape == y.shape)
    assert (y.shape[0] == x.shape[0] and y.shape[1] == x.shape[1])
    # initialize dl_dw
    h, w, c1, c2 = w_conv.shape
    H, W, c1 = x.shape
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)
    # pad the H and W axis not the C axis with appropriate number of zeros as determined by h and w axes of filter
    padding_h = int(h / 2)
    padding_w = int(w / 2)
    padded_x = np.pad(x, ((padding_h, padding_h), (padding_w, padding_w), (0, 0)), mode='constant')

    for c2_i in range(c2):
        L = dl_dy[:, :, c2_i]
        dl_db[c2_i] = np.sum(L)
        for c1_i in range(c1):
            # note: padding_h is also equal to step(h / 2)
            # note: padding_w is also equal to step(w / 2)
            for hi in range(h):
                for wi in range(w):
                    X = padded_x[hi: hi + H, wi:wi + W, c1_i]
                    dl_dw[hi, wi, c1_i, c2_i] = np.sum(np.multiply(L, X))

    return dl_dw, dl_db


def pool2x2(x):
    h, w, c = x.shape
    y = np.zeros((int(h / 2), int(w / 2), c))
    # remove last row and column from consideration if odd
    if h % 2 == 1:
        h -= 1
    if w % 2 == 1:
        w -= 1
    for ci in range(c):
        channel = x[:, :, ci]
        for hi in range(0, h, 2):
            for wi in range(0, w, 2):
                patch = channel[hi: hi + 2, wi:  wi + 2]
                # pick max value from patch and put it in y
                y[int(hi / 2), int(wi / 2), ci] = np.max(patch)
    return y


def pool2x2_backward(dl_dy, x, y):
    dl_dx = np.zeros(x.shape)
    h, w, c = x.shape
    # remove last row and column from consideration if odd
    if h % 2 == 1:
        h -= 1
    if w % 2 == 1:
        w -= 1
    for ci in range(c):
        channel = x[:, :, ci]
        for hi in range(0, h, 2):
            for wi in range(0, w, 2):
                patch = channel[hi: hi + 2, wi:  wi + 2]
                # pick max value's position from patch and put corresponding dl_dy in it
                max_pos = np.argmax(patch)
                dl_dx[hi + int(max_pos / 2), wi + max_pos % 2, ci] = dl_dy[int(hi / 2), int(wi / 2), ci]
    return dl_dx


def flattening(x):
    y = x.reshape(-1, order='F')
    y = y.reshape((y.shape[0], 1))
    return y


def flattening_backward(dl_dy, x, y):
    dl_dx = dl_dy.reshape(x.shape, order='F')
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
        if (iter_i + 1) % 1000 == 0:
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
        if (iter_i + 1) % 1000 == 0:
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
    # Constant hyper-parameters
    OUTPUT_SIZE = 10
    MIDDLE_LAYER_SIZE = 30
    INPUT_SIZE = 196
    # Tunable hyper-parameters
    LEARNING_RATE = 0.02
    DECAY_RATE = 0.9
    DECAY_PER_NUM_ITER = 500
    NUM_ITERATIONS = 18170
    TEST_ITER = 10
    global RELU_E
    RELU_E = 0.9
    print('MLP config: '
          'LEARNING_RATE = {}, '
          'DECAY_RATE = {}, '
          'DECAY_PER_NUM_ITER = {}, '
          'NUM_ITERATIONS = {}, '
          'RELU_E = {} '
          'TEST_ITER = {} '
          'OUTPUT_SIZE = {}, '
          'MIDDLE_LAYER_SIZE = {}, '
          'INPUT_SIZE = {} '.format(LEARNING_RATE,
                                    DECAY_RATE,
                                    DECAY_PER_NUM_ITER,
                                    NUM_ITERATIONS,
                                    RELU_E,
                                    TEST_ITER,
                                    OUTPUT_SIZE,
                                    MIDDLE_LAYER_SIZE,
                                    INPUT_SIZE, ))
    # weights = sio.loadmat('./mlp.mat')
    # w1 = weights['w1']
    # b1 = weights['b1']
    # w2 = weights['w2']
    # b2 = weights['b2']
    w1 = np.random.normal(0, 1, size=(MIDDLE_LAYER_SIZE, INPUT_SIZE))
    b1 = np.random.normal(0, 1, size=(MIDDLE_LAYER_SIZE, 1))
    w2 = np.random.normal(0, 1, size=(OUTPUT_SIZE, MIDDLE_LAYER_SIZE))
    b2 = np.random.normal(0, 1, size=(OUTPUT_SIZE, 1))
    num_mini_batches = len(mini_batches_x)
    loss_values = []
    latest_accuracy = 0
    best_accuracy = 0
    best_accuracy_iter = 0
    for iter_i in range(NUM_ITERATIONS):
        # print('Iteration # {}/{}: latest_accuracy = {}, best_accuracy = {} best_accuracy_iter = {} \r'.format(iter_i,
        #                                                                                                       NUM_ITERATIONS,
        #                                                                                                       latest_accuracy,
        #                                                                                                       best_accuracy,
        #                                                                                                       best_accuracy_iter),
        #       end='')
        print('Iteration # {}/{}: \r'.format(iter_i, NUM_ITERATIONS), end='')
        if (iter_i + 1) % DECAY_PER_NUM_ITER == 0:
            LEARNING_RATE = LEARNING_RATE * DECAY_RATE
        # if iter_i % TEST_ITER == 0:
        #     acc = 0
        #     confusion = np.zeros((10, 10))
        #     num_test = im_test.shape[1]
        #     for i in range(num_test):
        #         x = im_test[:, [i]]
        #         pred1 = fc(x, w1, b1)
        #         pred2 = relu(pred1)
        #         y = fc(pred2, w2, b2)
        #         l_pred = np.argmax(y)
        #         confusion[l_pred, label_test[0, i]] = confusion[l_pred, label_test[0, i]] + 1
        #
        #         if l_pred == label_test[0, i]:
        #             acc = acc + 1
        #     accuracy = acc / num_test
        #     latest_accuracy = accuracy
        #     if latest_accuracy > best_accuracy:
        #         best_accuracy = latest_accuracy
        #         best_accuracy_iter = iter_i
        #     if accuracy >= 0.90:
        #         return w1, b1, w2, b2

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
            # Forward prop
            # FC 1
            fc_x = fc(x.reshape(INPUT_SIZE, 1), w1, b1).reshape(-1)
            # ReLU 1
            relu_fc_x = relu(fc_x)
            # FC 2
            fc_relu_fc_x = fc(relu_fc_x.reshape(MIDDLE_LAYER_SIZE, 1), w2, b2).reshape(-1)

            # Loss
            l, dl_dy = loss_cross_entropy_softmax(fc_relu_fc_x, y)
            mini_batch_loss += np.abs(l)

            # Back prop
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
        # print(np.mean(w1), np.mean(b1), np.mean(w2), np.mean(b2))

    print()
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.xlabel('iterations', fontsize=18)
    plt.ylabel('training loss', fontsize=16)
    plt.plot(loss_values)
    plt.show()

    return w1, b1, w2, b2


def train_cnn(mini_batches_x, mini_batches_y
              #, im_test, label_test
              ):
    # Constant hyper-parameters
    W_CONV_SHAPE = (3, 3, 1, 3)
    B_CONV_SHAPE = 3
    OUTPUT_SIZE = 10
    PEN_ULTIMATE_SIZE = 147
    INPUT_SIZE = 196
    # Tunable hyper-parameters
    global RELU_E
    RELU_E = 0.5
    LEARNING_RATE = 0.002
    NUM_ITERATIONS = 7500
    DECAY_RATE = 0.6
    DECAY_PER_NUM_ITER = 500
    TEST_ITER = 500
    CONFIG = 'CNN config: (LEARNING_RATE, DECAY_RATE, DECAY_PER_NUM_ITER, NUM_ITERATIONS, RELU_E, W_CONV_SHAPE, B_CONV_SHAPE, OUTPUT_SIZE, PEN_ULTIMATE_SIZE, INPUT_SIZE, TEST_ITER) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
        LEARNING_RATE,
        DECAY_RATE,
        DECAY_PER_NUM_ITER,
        NUM_ITERATIONS,
        RELU_E,
        W_CONV_SHAPE,
        B_CONV_SHAPE,
        OUTPUT_SIZE,
        PEN_ULTIMATE_SIZE,
        INPUT_SIZE,
        TEST_ITER)
    print(CONFIG)
    w_conv = np.random.normal(0, 1, size=W_CONV_SHAPE)
    b_conv = np.random.normal(0, 1, size=B_CONV_SHAPE)
    w_fc = np.random.normal(0, 1, size=(OUTPUT_SIZE, PEN_ULTIMATE_SIZE))
    b_fc = np.random.normal(0, 1, size=(OUTPUT_SIZE, 1))
    num_mini_batches = len(mini_batches_x)
    loss_values = []
    for iter_i in range(NUM_ITERATIONS):
        if (iter_i + 1) % DECAY_PER_NUM_ITER == 0:
            LEARNING_RATE = LEARNING_RATE * DECAY_RATE
        # if (iter_i + 1) % TEST_ITER == 0:
        #     acc = 0
        #     confusion = np.zeros((10, 10))
        #     num_test = im_test.shape[1]
        #     print()
        #     for i in range(num_test):
        #         print('Test # {}/{}: \r'.format(i + 1, num_test), end='')
        #         x = im_test[:, [i]].reshape((14, 14, 1), order='F')
        #         pred1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
        #         pred2 = relu(pred1)  # (14, 14, 3)
        #         pred3 = pool2x2(pred2)  # (7, 7, 3)
        #         pred4 = flattening(pred3)  # (147, 1)
        #         y = fc(pred4, w_fc, b_fc)  # (10, 1)
        #         l_pred = np.argmax(y)
        #         confusion[l_pred, label_test[0, i]] = confusion[l_pred, label_test[0, i]] + 1
        #         if l_pred == label_test[0, i]:
        #             acc = acc + 1
        #     accuracy = acc / num_test
        #     print()
        #     print(accuracy)

        # Determining current mini-batch
        curr_mini_batch_x = mini_batches_x[iter_i % num_mini_batches]
        curr_mini_batch_y = mini_batches_y[iter_i % num_mini_batches]
        curr_mini_batch_size = curr_mini_batch_x.shape[1]
        # Current mini-batch gradients
        mini_batch_dl_dw_conv = np.zeros(W_CONV_SHAPE)
        mini_batch_dl_db_conv = np.zeros(B_CONV_SHAPE)
        mini_batch_dl_dw_fc = np.zeros(OUTPUT_SIZE * PEN_ULTIMATE_SIZE)
        mini_batch_dl_db_fc = np.zeros(OUTPUT_SIZE)
        mini_batch_loss = 0
        for idx in range(curr_mini_batch_size):
            x = curr_mini_batch_x[:, idx].reshape((14, 14, 1), order='F')
            y = curr_mini_batch_y[:, idx]
            # Forward prop
            # Conv 1
            conv_x = conv(x, w_conv, b_conv)
            # ReLU 1
            relu_conv_x = relu(conv_x)
            # Pool 1
            pool_relu_conv_x = pool2x2(relu_conv_x)
            # Flatten 1
            flatten_pool_relu_conv_x = flattening(pool_relu_conv_x)
            # FC 1
            fc_flatten_pool_relu_conv_x = fc(flatten_pool_relu_conv_x, w_fc, b_fc)

            # Loss
            l, dl_dy = loss_cross_entropy_softmax(fc_flatten_pool_relu_conv_x.reshape(-1), y)
            mini_batch_loss += np.abs(l)

            # Back prop
            # FC 1
            dl_dy, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, flatten_pool_relu_conv_x, w_fc, b_fc,
                                                    fc_flatten_pool_relu_conv_x)
            # Flatten 1
            dl_dy = flattening_backward(dl_dy, pool_relu_conv_x, flatten_pool_relu_conv_x)
            # Pool 1
            dl_dy = pool2x2_backward(dl_dy, relu_conv_x, pool_relu_conv_x)
            # ReLu 1
            dl_dy = relu_backward(dl_dy, conv_x, relu_conv_x)
            # Conv 1
            dl_dw_conv, dl_db_conv = conv_backward(dl_dy, x, w_conv, b_conv, conv_x)

            # Sum gradients
            mini_batch_dl_dw_conv += dl_dw_conv
            # mini_batch_dl_db_conv += dl_db_conv
            mini_batch_dl_dw_fc += dl_dw_fc
            mini_batch_dl_db_fc += dl_db_fc

        print('Iteration # {}/{} loss = {} \r'.format(iter_i + 1, NUM_ITERATIONS, mini_batch_loss), end='')
        loss_values.append(mini_batch_loss)
        # Update
        # print(np.mean(np.abs(mini_batch_dl_dw_conv)), np.mean(np.abs(mini_batch_dl_db_conv)), np.mean(np.abs(mini_batch_dl_dw_fc)), np.mean(np.abs(mini_batch_dl_db_fc)))
        w_conv = w_conv - mini_batch_dl_dw_conv * LEARNING_RATE
        # b_conv = b_conv - mini_batch_dl_db_conv * LEARNING_RATE
        w_fc = w_fc - mini_batch_dl_dw_fc.reshape(w_fc.shape) * LEARNING_RATE
        b_fc = b_fc - mini_batch_dl_db_fc.reshape(b_fc.shape) * LEARNING_RATE

    print()
    axes = plt.gca()
    plt.xlabel('iterations', fontsize=18)
    plt.ylabel('training loss', fontsize=16)
    plt.plot(loss_values)
    plt.show()

    return w_conv, b_conv, w_fc, b_fc

if __name__ == '__main__':
    np.random.seed(42)
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()
