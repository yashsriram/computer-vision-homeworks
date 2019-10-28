import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(
            PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


def compute_dsift(img):
    # To do
    return dense_feature


def get_tiny_image(img, output_size):
    feature = cv2.resize(img, output_size)
    # cv2.imshow('tiny_image', feature)
    # cv2.waitKey(5000)
    feature = feature.reshape(-1)
    norm = np.linalg.norm(feature)
    feature = feature / norm
    mean = np.mean(feature)
    feature = feature - mean
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(feature_train, label_train)
    test_labels = neigh.predict(feature_test)
    return test_labels


def classify_knn_tiny(class_labels, train_labels, train_images, true_test_labels, test_images):
    TINY_FEATURE_SIZE = (16, 16)
    K_IN_KNN = 7
    # Create train features
    feature_train = []
    for img_file in train_images:
        img = cv2.imread(img_file, 0)
        tiny_img = get_tiny_image(img, TINY_FEATURE_SIZE)
        feature_train.append(tiny_img)
    feature_train = np.asarray(feature_train)

    # Create test features
    feature_test = []
    for img_file in test_images:
        img = cv2.imread(img_file, 0)
        tiny_img = get_tiny_image(img, TINY_FEATURE_SIZE)
        feature_test.append(tiny_img)
    feature_test = np.asarray(feature_test)

    predicted_test_labels = predict_knn(feature_train, train_labels, feature_test, K_IN_KNN)
    confusion = confusion_matrix(true_test_labels, predicted_test_labels)
    accuracy = accuracy_score(true_test_labels, predicted_test_labels)
    print('TINY+KNN : tiny_size: {} kNN\'s k: {} accuracy: {}'.format(TINY_FEATURE_SIZE, K_IN_KNN, accuracy))

    visualize_confusion_matrix(confusion, accuracy, class_labels)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    return vocab


def compute_bow(feature, vocab):
    # To do
    return bow_feature


def classify_knn_bow(class_labels, train_labels, train_images, true_test_labels, test_images):
    # To do
    visualize_confusion_matrix(confusion, accuracy, class_labels)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    return label_test_pred


def classify_svm_bow(class_labels, train_labels, train_images, true_test_labels, test_images):
    # To do
    visualize_confusion_matrix(confusion, accuracy, class_labels)
    return confusion, accuracy


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list \
        = extract_dataset_info("./scene_classification_data")

    _, accuracy \
        = classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
