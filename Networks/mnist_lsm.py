import numpy as np
from tqdm import tqdm
import sys
import loadDataset

import feedforward
import reservoir
import svm


def lsm(n_inputs, n_classes, n_steps, epoches, x_train, x_test, y_train, y_test, classifier, data_set):
    dim1 = [10, 10, 20]
    r1 = reservoir.ReservoirLayer(n_inputs, 2000, n_steps, dim1, is_input=True, homeostasis=True)
    s1 = feedforward.SpikingLayer(2000, n_classes, n_steps)
    accuracy = 0
    best_acc_e = 0

    if classifier == "calcium_supervised":
        for e in range(epoches):
            record = np.zeros(n_classes)
            for i in tqdm(range(len(x_train))):  # train phase
                r1.reset()
                s1.reset()
                if data_set == "TI46":
                    x = np.asarray(x_train[i].todense())
                elif data_set == "MNIST" and record[y_train[i]] < 80:
                    x = loadDataset.genrate_poisson_spikes(x_train[i], n_steps)
                    record[y_train[i]] += 1
                else:
                    continue
                o_r1 = r1.forward(x)
                s1.forward(o_r1, e, y_train[i])

            correct = 0
            num_test = 0
            for i in tqdm(range(len(x_test))):  # test phase
                r1.reset()
                s1.reset()
                if data_set == "TI46":
                    x = np.asarray(x_test[i].todense())
                elif data_set == "MNIST" and record[y_test[i]] < 100:
                    x = loadDataset.genrate_poisson_spikes(x_test[i], n_steps)
                    record[y_test[i]] += 1
                else:
                    continue
                o_r1 = r1.forward(x)
                o_s1 = s1.forward(o_r1)

                fire_count = np.sum(o_s1, axis=0)
                # print(fire_count)
                index = np.argmax(fire_count)
                if index == y_test[i]:
                    correct = correct + 1
                num_test += 1
            acc = correct / num_test
            print("test accuracy at epoch %d is %0.2f%%" % (e, acc * 100))
            if accuracy < acc:
                accuracy = acc
                best_acc_e = e
    elif classifier == "svmcv":
        samples = []
        label = []
        record = np.zeros(n_classes)
        for i in tqdm(range(len(x_train))):  # train phase
            r1.reset()
            s1.reset()
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
