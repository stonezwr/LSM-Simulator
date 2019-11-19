import numpy as np
import random
import matplotlib.pyplot as plot


def plot_v(v, num):
    for i in range(num):
        f = plot.plot(v[:, i])
    plot.show(f)


class SpikingLayer:
    def __init__(self, n_inputs, n_outputs, n_steps, tau_m=64, tau_s=8, tau_c=64, cal_mid=5, cal_margin=3,
                 threshold=15, refrac=2, weight_scale=8, weight_limit=8, is_input=False, n_input_connect=32,
                 delta_pot=0.006, delta_dep=0.006, dtype=np.float32):
        self.dtype = dtype
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.tau_c = tau_c
        self.threshold = threshold
        self.refrac = refrac
        self.weight_scale = weight_scale
        self.n_steps = n_steps
        self.cal_mid = cal_mid
        self.cal_margin = cal_margin
        self.delta_pot = delta_pot
        self.delta_dep = delta_dep
        self.weight_limit = weight_limit
        self.w = np.zeros((self.n_inputs, self.n_outputs), dtype=self.dtype)
        self.init_input(n_input_connect, is_input)

        self.v = np.zeros(self.n_outputs, dtype=self.dtype)
        self.syn = np.zeros(self.n_outputs, dtype=self.dtype)
        self.cal = np.zeros(self.n_outputs, dtype=self.dtype)

    def reset(self):
        self.v = np.zeros(self.n_outputs, dtype=self.dtype)
        self.syn = np.zeros(self.n_outputs, dtype=self.dtype)

    def init_input(self, num, is_input):
        if is_input:
            for pre in range(self.n_inputs):
                for j in range(num):
                    post = random.randrange(self.n_outputs)
                    self.w[pre, post] = self.w[pre, post] + random.uniform(-1, 1) * self.weight_scale
        else:
            self.w = np.random.rand(self.n_inputs, self.n_outputs)
            self.w = self.w * 2 - 1

    def forward(self, inputs, epoch=-1, label=-1):
        h1 = np.matmul(inputs, self.w)
        outputs = []
        v_all = []
        cal_all = []
        ref = np.zeros(self.n_outputs)
        for t in range(self.n_steps):
            self.syn = self.syn - self.syn / self.tau_s + h1[t, :]
            self.v = self.v - self.v / self.tau_m + self.syn / self.tau_s
            self.cal = self.cal - self.cal/self.tau_c
            if label >= 0:
                teacher = np.zeros(self.n_outputs, dtype=np.int)
                teacher[label] = 1
                self.v[(teacher == 1) & (self.cal < (self.cal_mid + 1))] = \
                    self.v[(teacher == 1) & (self.cal < (self.cal_mid + 1))] + self.threshold
                self.v[(teacher == 0) & (self.cal > (self.cal_mid - 1))] = \
                    self.v[(teacher == 0) & (self.cal > (self.cal_mid - 1))] - self.threshold * 0.75

            self.v[ref > 0] = 0
            ref[ref > 0] = ref[ref > 0] - 1
            v_all.append(self.v)

            out = np.zeros(self.n_outputs, dtype=self.dtype)
            out[self.v > self.threshold] = 1.0
            outputs.append(out)

            self.cal[self.v > self.threshold] = self.cal[self.v > self.threshold] + 1
            cal_all.append(self.cal)
            if label >= 0:
                self.calcuim_supervised_rule(epoch, np.asarray(inputs[t, :]))
            self.v[self.v > self.threshold] = 0
            ref[self.v > self.threshold] = self.refrac
        outputs = np.stack(outputs)
        v_all = np.stack(v_all)
        cal_all = np.stack(cal_all)
        return outputs

    def calcuim_supervised_rule(self, epoch, inputs):

        mask = np.zeros((self.n_inputs, self.n_outputs), dtype=np.int)
        mask[inputs == 1, :] = mask[inputs == 1, :] + 1
        mask[:, (self.cal_mid < self.cal) & (self.cal < (self.cal_mid + self.cal_margin))] = \
            mask[:, (self.cal_mid < self.cal) & (self.cal < (self.cal_mid + self.cal_margin))] + 1
        val = self.delta_pot / (1 + epoch / 25)
        self.w[mask == 2] = self.w[mask == 2] + val

        mask = np.zeros((self.n_inputs, self.n_outputs), dtype=np.int)
        mask[inputs == 1, :] = mask[inputs == 1, :] + 1
        mask[:, ((self.cal_mid - self.cal_margin) < self.cal) & (self.cal < self.cal_mid)] = \
            mask[:, ((self.cal_mid - self.cal_margin) < self.cal) & (self.cal < self.cal_mid)] + 1

        val = self.delta_dep / (1 + epoch / 25)
        self.w[mask == 2] = self.w[mask == 2] - val

        self.w[self.w > self.weight_limit] = self.weight_limit
        self.w[self.w < -self.weight_limit] = -self.weight_limit

        # for i in range(self.n_inputs):
        #    if inputs[i] == 0:
        #        continue
        # for j in range(self.n_outputs):
        #    if self.cal_mid < self.cal[j] < (self.cal_mid + self.cal_margin):
        #        self.w[i, j] = self.w[i, j] + self.delta_pot / (1 + tmp / 25)
        #    if (self.cal_mid - self.cal_margin) < self.cal < self.cal_mid:
        #        self.w[i, j] = self.w[i, j] - self.delta_pot / (1 + tmp / 25)