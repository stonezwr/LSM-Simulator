import numpy as np
from tqdm import tqdm
import sys
import loadDataset

import feedforward
import reservoir
import svm


def lsm(n_inputs, n_classes, n_steps, epoches, x_train, x_test, y_train, y_test, classifier, data_set):
    stdp = False  # stdp enabled
    dim1 = [3, 3, 15]
    r1 = reservoir.ReservoirLayer(n_inputs, 135, n_steps, dim1, is_input=True, homeostasis=True)
    s1 = feedforward.SpikingLayer(135, n_classes, n_steps, homeostasis=True)
    accuracy = 0
    best_acc_e = 0
    
    # train stdp
    if stdp:
        # r1.stdp_i = True
        r1.stdp_r = True
        # s1.stdp_i = True
        print("start stdp")
        for e_stdp in range(5):
            for i in tqdm(range(len(x_train))):
                r1.reset()
                s1.reset()
                x = np.asarray(x_train[i].todense())
                o_r1 = r1.forward(x)
                s1.forward(o_r1)
        print("finish stdp")
        r1.stdp_i = False
        r1.stdp_r = False
        s1.stdp_i = False

    if classifier == "calcium_supervised":
        for e in range(epoches):
            for i in tqdm(range(len(x_train))):  # train phase
                r1.reset()
                s1.reset()
                x = np.asarray(x_train[i].todense())
                o_r1 = r1.forward(x)
                s1.forward(o_r1, e, y_train[i])

            correct = 0
            for i in tqdm(range(len(x_test))):  # test phase
                r1.reset()
                s1.reset()
                x = np.asarray(x_test[i].todense())
                o_r1 = r1.forward(x)
                o_s1 = s1.forward(o_r1)

                fire_count = np.sum(o_s1, axis=0)
                index = np.argmax(fire_count)
                if index == y_test[i]:
                    correct = correct + 1
            acc = correct / len(x_test)
            print("test accuracy at epoch %d is %0.2f%%" % (e, acc * 100))
            if accuracy < acc:
                accuracy = acc
                best_acc_e = e
    elif classifier == "svm":
        train_samples = []
        test_samples = []
        for i in tqdm(range(len(x_train))):  # train phase
            r1.reset()
            x = np.asarray(x_train[i].todense())
            o_r1 = r1.forward(x)
            fire_count = np.sum(o_r1, axis=0)
            train_samples.append(fire_count)

        for i in tqdm(range(len(x_test))):
            r1.reset()
            x = np.asarray(x_test[i].todense())
            o_r1 = r1.forward(x)
            fire_count = np.sum(o_r1, axis=0)
            test_samples.append(fire_count)

        accuracy = svm.traintestSVM(train_samples, y_train, test_samples, y_test)
        best_acc_e = 0
    elif classifier == "svmcv":
        samples = []
        label = []
        record = np.zeros(n_classes)
        for i in tqdm(range(len(x_train))):
            r1.reset()
            if data_set == "TI46":
                x = np.asarray(x_train[i].todense())
            elif data_set == "MNIST" and record[y_train[i]] < 100:
                x = loadDataset.genrate_poisson_spikes(x_train[i], n_steps)
                record[y_train[i]] += 1
            else:
                continue
            o_r1 = r1.forward(x)
            fire_count = np.sum(o_r1, axis=0)
            samples.append(fire_count)
            label.append(y_train[i])

        accuracy = svm.cvSVM(samples, label, 5)
        best_acc_e = 0
    else:
        print('Given classifier {} not found'.format(classifier))
        sys.exit(-1)
    return accuracy, best_acc_e
