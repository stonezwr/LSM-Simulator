import os
import sys
import loadDataset
import network
import numpy as np
import random


def shuffle_dataset(data, label):
    c = list(zip(data, label))
    random.shuffle(c)
    data, label = zip(*c)
    return data, label


if __name__ == "__main__":
    n_steps = 750
    n_channels = 78
    n_classes = 26
    epoches = 150
    data_set = "TI46"

    # n_steps = 400
    # n_channels = 784
    # n_classes = 10
    # data_set = "MNIST"

    classifier = "calcium_supervised"
    # classifier = "svm"
    # classifier = "svmcv"

    dtype = np.float32

    if data_set == "MNIST":
        data_path = os.path.expanduser("./mnist")

        trainset, testset, classes = loadDataset.get_mnist(data_path)

        # Standardize data
        x_train = np.array(trainset.train_data, dtype=np.float)
        x_train = x_train.reshape(x_train.shape[0], -1)/255

        x_test = np.array(testset.test_data, dtype=np.float)
        x_test = x_test.reshape(x_test.shape[0], -1)/255

        y_train = np.array(trainset.targets, dtype=np.int)
        y_test = np.array(testset.targets, dtype=np.int)

    elif data_set == "TI46":
        speaker_per_class = 1
        data_path = os.path.expanduser("./TI46_alpha")
        x_train, x_test, y_train, y_test = loadDataset.loadTI46Alpha(data_path, speaker_per_class, n_steps, n_channels, classifier)
    else:
        print("please choose correct dataset, MNIST or TI46")
        sys.exit(-1)

    if classifier == "calcium_supervised":
        x_train, y_train = shuffle_dataset(x_train, y_train)

    accuracy, e = network.lsm(n_channels, n_classes, n_steps, epoches, x_train, x_test, y_train, y_test, classifier, data_set)
    print("best accuracy: %0.2f%% is achieved at epoch %d" % (accuracy*100, e))


