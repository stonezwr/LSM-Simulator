import numpy as np
from tqdm import tqdm
import sys
import loadDataset

import feedforward
import reservoir
import svm


def reset(layers):
    for i in range(len(layers)):
        layers[i].reset()


def lsm(n_inputs, n_classes, n_steps, epoches, x_train, x_test, y_train, y_test, classifier, data_set):
    dim1 = [5, 5, 4]
    layers = []
    r1 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)
    r2 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)
    r3 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)
    r4 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)
    r5 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)
    r6 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)
    r7 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)
    r8 = reservoir.ReservoirLayer(int(n_inputs/8), 100, n_steps, dim1, is_input=True, homeostasis=True)

    s1 = feedforward.SpikingLayer(200, 150, n_steps, homeostasis=True)
    s2 = feedforward.SpikingLayer(200, 150, n_steps, homeostasis=True)
    s3 = feedforward.SpikingLayer(200, 150, n_steps, homeostasis=True)
    s4 = feedforward.SpikingLayer(200, 150, n_steps, homeostasis=True)

    s5 = feedforward.SpikingLayer(300, 150, n_steps, homeostasis=True)
    s6 = feedforward.SpikingLayer(300, 150, n_steps, homeostasis=True)

    s7 = feedforward.SpikingLayer(300, 200, n_steps, homeostasis=True)
    s8 = feedforward.SpikingLayer(200, n_classes, n_steps)

    layers.append(r1)
    layers.append(r2)
    layers.append(r3)
    layers.append(r4)
    layers.append(r5)
    layers.append(r6)
    layers.append(r7)
    layers.append(r8)
    layers.append(s1)
    layers.append(s2)
    layers.append(s3)
    layers.append(s4)
    layers.append(s5)
    layers.append(s6)
    layers.append(s7)
    layers.append(s8)

    accuracy = 0
    best_acc_e = 0

    if classifier == "calcium_supervised":
        for e in range(epoches):
            record = np.zeros(n_classes)
            for i in tqdm(range(len(x_train))):  # train phase
                reset(layers)
                if data_set == "TI46":
                    x = np.asarray(x_train[i].todense())
                elif data_set == "MNIST" and record[y_train[i]] < 80:
                    x = loadDataset.genrate_poisson_spikes(x_train[i], n_steps)
                    record[y_train[i]] += 1
                else:
                    continue
                o_r1 = r1.forward(x[:, int(0 * n_inputs / 8): int(1 * n_inputs / 8)])
                o_r2 = r2.forward(x[:, int(1 * n_inputs / 8): int(2 * n_inputs / 8)])
                o_r3 = r3.forward(x[:, int(2 * n_inputs / 8): int(3 * n_inputs / 8)])
                o_r4 = r4.forward(x[:, int(3 * n_inputs / 8): int(4 * n_inputs / 8)])
                o_r5 = r5.forward(x[:, int(4 * n_inputs / 8): int(5 * n_inputs / 8)])
                o_r6 = r6.forward(x[:, int(5 * n_inputs / 8): int(6 * n_inputs / 8)])
                o_r7 = r7.forward(x[:, int(6 * n_inputs / 8): int(7 * n_inputs / 8)])
                o_r8 = r8.forward(x[:, int(7 * n_inputs / 8): int(8 * n_inputs / 8)])

                o_s1 = s1.forward(np.concatenate((o_r1, o_r2), 1))
                o_s2 = s2.forward(np.concatenate((o_r3, o_r4), 1))
                o_s3 = s3.forward(np.concatenate((o_r5, o_r6), 1))
                o_s4 = s4.forward(np.concatenate((o_r7, o_r8), 1))

                o_s5 = s5.forward(np.concatenate((o_s1, o_s2), 1))
                o_s6 = s6.forward(np.concatenate((o_s3, o_s4), 1))

                o_s7 = s7.forward(np.concatenate((o_s5, o_s6), 1))

                s1.forward(o_s7, e, y_train[i])

            correct = 0
            for i in tqdm(range(len(x_test))):  # test phase
                reset(layers)
                if data_set == "TI46":
                    x = np.asarray(x_test[i].todense())
                elif data_set == "MNIST" and record[y_test[i]] < 100:
                    x = loadDataset.genrate_poisson_spikes(x_test[i], n_steps)
                    record[y_test[i]] += 1
                else:
                    continue
                o_r1 = r1.forward(x[:, int(0 * n_inputs / 8): int(1 * n_inputs / 8)])
                o_r2 = r2.forward(x[:, int(1 * n_inputs / 8): int(2 * n_inputs / 8)])
                o_r3 = r3.forward(x[:, int(2 * n_inputs / 8): int(3 * n_inputs / 8)])
                o_r4 = r4.forward(x[:, int(3 * n_inputs / 8): int(4 * n_inputs / 8)])
                o_r5 = r5.forward(x[:, int(4 * n_inputs / 8): int(5 * n_inputs / 8)])
                o_r6 = r6.forward(x[:, int(5 * n_inputs / 8): int(6 * n_inputs / 8)])
                o_r7 = r7.forward(x[:, int(6 * n_inputs / 8): int(7 * n_inputs / 8)])
                o_r8 = r8.forward(x[:, int(7 * n_inputs / 8): int(8 * n_inputs / 8)])

                o_s1 = s1.forward(np.concatenate((o_r1, o_r2), 1))
                o_s2 = s2.forward(np.concatenate((o_r3, o_r4), 1))
                o_s3 = s3.forward(np.concatenate((o_r5, o_r6), 1))
                o_s4 = s4.forward(np.concatenate((o_r7, o_r8), 1))

                o_s5 = s5.forward(np.concatenate((o_s1, o_s2), 1))
                o_s6 = s6.forward(np.concatenate((o_s3, o_s4), 1))

                o_s7 = s7.forward(np.concatenate((o_s5, o_s6), 1))

                o_s8 = s1.forward(o_s7)

                fire_count = np.sum(o_s8, axis=0)
                # print(fire_count)
                index = np.argmax(fire_count)
                if index == y_test[i]:
                    correct = correct + 1
            acc = correct / len(x_test)
            print("test accuracy at epoch %d is %0.2f%%" % (e, acc * 100))
            if accuracy < acc:
                accuracy = acc
                best_acc_e = e

    elif classifier == "svmcv":
        samples = []
        label = []
        record = np.zeros(n_classes)
        for i in tqdm(range(len(x_train))):  # train phase
            reset(layers)
            if data_set == "TI46":
                x = np.asarray(x_train[i].todense())
            elif data_set == "MNIST" and record[y_train[i]] < 100:
                x = loadDataset.genrate_poisson_spikes(x_train[i], n_steps)
                record[y_train[i]] += 1
            else:
                continue
            o_r1 = r1.forward(x[:, int(0 * n_inputs / 8): int(1 * n_inputs / 8)])
            o_r2 = r2.forward(x[:, int(1 * n_inputs / 8): int(2 * n_inputs / 8)])
            o_r3 = r3.forward(x[:, int(2 * n_inputs / 8): int(3 * n_inputs / 8)])
            o_r4 = r4.forward(x[:, int(3 * n_inputs / 8): int(4 * n_inputs / 8)])
            o_r5 = r5.forward(x[:, int(4 * n_inputs / 8): int(5 * n_inputs / 8)])
            o_r6 = r6.forward(x[:, int(5 * n_inputs / 8): int(6 * n_inputs / 8)])
            o_r7 = r7.forward(x[:, int(6 * n_inputs / 8): int(7 * n_inputs / 8)])
            o_r8 = r8.forward(x[:, int(7 * n_inputs / 8): int(8 * n_inputs / 8)])

            o_s1 = s1.forward(np.concatenate((o_r1, o_r2), 1))
            o_s2 = s2.forward(np.concatenate((o_r3, o_r4), 1))
            o_s3 = s3.forward(np.concatenate((o_r5, o_r6), 1))
            o_s4 = s4.forward(np.concatenate((o_r7, o_r8), 1))

            o_s5 = s5.forward(np.concatenate((o_s1, o_s2), 1))
            o_s6 = s6.forward(np.concatenate((o_s3, o_s4), 1))

            o_s7 = s7.forward(np.concatenate((o_s5, o_s6), 1))
            fire_count = np.sum(o_s7, axis=0)
            samples.append(fire_count)
            label.append(y_train[i])

        accuracy = svm.cvSVM(samples, label, 5)
        best_acc_e = 0
    else:
        print('Given classifier {} not found'.format(classifier))
        sys.exit(-1)
    return accuracy, best_acc_e
