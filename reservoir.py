import numpy as np
import random
import math


def dist(p1, p2):
    d = p1 - p2
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


class ReservoirLayer:
    def __init__(self, n_inputs, n_outputs, n_steps, dim, tau_m=64, tau_s=8,
                 threshold=10, refrac=2, weight_scale=8, weight_limit=8, is_input=False,
                 n_input_connect=32, homeostasis=False, stdp_r=False, stdp_i=False, dtype=np.float32):
        self.dtype = dtype
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dim = dim
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.threshold = np.ones(self.n_outputs, dtype=self.dtype) * threshold
        self.refrac = refrac
        self.weight_scale = weight_scale
        self.weight_limit = weight_limit
        self.n_steps = n_steps
        self.homeostasis = homeostasis
        self.stdp_r = stdp_r
        self.stdp_i = stdp_i
        self.stdp_lambda = 1/512
        self.stdp_TAU_X_TRACE_E = 4
        self.stdp_TAU_X_TRACE_I = 2
        self.stdp_TAU_Y_TRACE_E = 8
        self.stdp_TAU_Y_TRACE_I = 4
        self.A_neg = 0.01
        self.A_pos = 0.005
        self.trace_x_i = np.zeros(self.n_inputs, dtype=self.dtype)
        self.trace_x_r = np.zeros(self.n_outputs, dtype=self.dtype)
        self.trace_y = np.zeros(self.n_outputs, dtype=self.dtype)

        self.excitatoty = np.random.rand(self.n_outputs)
        self.excitatoty[self.excitatoty < 0.2] = -1
        self.excitatoty[self.excitatoty >= 0.2] = 1
        self.w = np.zeros((self.n_inputs, self.n_outputs), dtype=self.dtype)
        self.init_input(n_input_connect, is_input)

        self.w_r = np.zeros((self.n_outputs, self.n_outputs), dtype=self.dtype)
        self.w_r[1, :] = 1

        self.v = np.zeros(self.n_outputs, dtype=self.dtype)
        self.syn = np.zeros(self.n_outputs, dtype=self.dtype)
        self.pre_out = np.zeros(self.n_outputs, dtype=self.dtype)

    def reset(self):
        self.v = np.zeros(self.n_outputs, dtype=self.dtype)
        self.syn = np.zeros(self.n_outputs, dtype=self.dtype)
        self.trace_x_i = np.zeros(self.n_inputs, dtype=self.dtype)
        self.trace_x_r = np.zeros(self.n_outputs, dtype=self.dtype)
        self.trace_y = np.zeros(self.n_outputs, dtype=self.dtype)
        self.pre_out = np.zeros(self.n_outputs, dtype=self.dtype)

    def init_input(self, num, is_input):
        if is_input:
            for pre in range(self.n_inputs):
                for j in range(num):
                    post = random.randrange(self.n_outputs)
                    self.w[pre, post] = self.w[pre, post] + random.uniform(-1, 1) * self.weight_scale
        else:
            self.w = np.random.rand(self.n_inputs, self.n_outputs)
            self.w = self.w * 2 - 1

    def init_reservoir(self):
        assert self.dim[0] * self.dim[1] * self.dim[2] == self.n_outputs
        p = []
        factor1 = 1.5
        factor2 = 4
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                for k in range(self.dim[2]):
                    p.append(np.asarray([i, j, k]))
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                if i == j:
                    continue
                if self.excitatoty[i] == 1:
                    if self.excitatoty[j] == 1:
                        prob = 0.3 * factor1
                        val = 1
                    else:
                        prob = 0.2 * factor1
                        val = 1
                else:
                    if self.excitatoty[j] == 1:
                        prob = 0.4 * factor1
                        val = -1
                    else:
                        prob = 0.2 * factor1
                        val = -1
                d = dist(p[i], p[j])
                r = random.random()
                if r < prob * math.exp(-d / factor2):
                    self.w_r[i][j] = val * 4

    def forward(self, inputs):
        h1 = np.matmul(inputs, self.w)
        outputs = []
        ref = np.zeros(self.n_outputs)
        for t in range(self.n_steps):
            h_r = np.matmul(self.pre_out, self.w_r)  # w_r: in*out
            self.syn = self.syn - self.syn / self.tau_s + (h1[t, :] + h_r) * self.excitatoty
            self.v = self.v - self.v / self.tau_m + self.syn / self.tau_s

            self.v[ref > 0] = 0
            ref[ref > 0] = ref[ref > 0] - 1
            v_thr = self.v - self.threshold
            out = np.zeros(self.n_outputs, dtype=self.dtype)
            out[v_thr > 0] = 1
            outputs.append(out)
            self.pre_out = out
            ref[v_thr > 0] = self.refrac
            self.v[v_thr > 0] = 0

            if self.homeostasis:
                self.threshold = self.threshold - self.threshold / 48
                self.threshold[out == 1] = self.threshold[out == 1] + 1
                self.threshold[self.threshold < 2] = 2
                self.threshold[self.threshold > 32] = 32

            if self.stdp_r or self.stdp_i:
                self.trace_y[self.excitatoty == 1] = self.trace_y[self.excitatoty == 1] / self.stdp_TAU_Y_TRACE_E
                self.trace_y[self.excitatoty == -1] = self.trace_y[self.excitatoty == -1] / self.stdp_TAU_Y_TRACE_I
                self.trace_y[out == 1] = self.trace_y[out == 1] + 1

            if self.stdp_r:
                self.trace_x_r[self.excitatoty == 1] = self.trace_x_r[self.excitatoty == 1] / self.stdp_TAU_X_TRACE_E
                self.trace_x_r[self.excitatoty == -1] = self.trace_x_r[self.excitatoty == -1] / self.stdp_TAU_X_TRACE_I
                self.trace_x_r[self.pre_out == 1] = self.trace_x_r[self.pre_out == 1] + 1

                m_y = np.tile(self.trace_y, (self.n_outputs, 1))
                w_tmp = self.A_neg * self.stdp_lambda * m_y
                w_tmp[self.w_r < 0] = -w_tmp[self.w_r < 0]
                self.w_r[self.pre_out == 1, :] = self.w_r[self.pre_out == 1, :] - w_tmp[self.pre_out == 1, :]

                m_x = np.tile(self.trace_x_r, (self.n_outputs, 1)).T
                w_tmp = self.A_pos * self.stdp_lambda * m_x
                w_tmp[self.w_r < 0] = -w_tmp[self.w_r < 0]
                self.w_r[:, out == 1] = self.w_r[:, out == 1] + w_tmp[:, out == 1]
                self.w_r[self.w_r > self.weight_limit] = self.weight_limit
                self.w_r[self.w_r < -self.weight_limit] = -self.weight_limit

            in_s = inputs[t, :]
            if self.stdp_i and (np.sum(in_s) > 0 or np.sum(out) > 0):
                in_s = inputs[t, :]
                self.trace_x_i[in_s == 1] = self.trace_x_i[in_s == 1] + 1
                self.trace_x_i = self.trace_x_i / self.stdp_TAU_X_TRACE_E
                m_y = np.tile(self.trace_y, (self.n_inputs, 1))
                # m_y = np.repeat(self.trace_y, self.n_inputs)
                # m_y = m_y.reshape((self.n_outputs, self.n_inputs))
                # m_y = m_y.T
                w_tmp = self.A_neg * self.stdp_lambda * m_y
                w_tmp[self.w < 0] = -w_tmp[self.w < 0]
                self.w[in_s == 1, :] = self.w[in_s == 1, :] - w_tmp[in_s == 1, :]
                m_x = np.tile(self.trace_x_r, (self.n_outputs, 1)).T
                # m_x = np.repeat(self.trace_x_i, self.n_outputs)
                # m_x = m_x.reshape((self.n_inputs, self.n_outputs))
                w_tmp = self.A_pos * self.stdp_lambda * m_x
                w_tmp[self.w < 0] = -w_tmp[self.w < 0]
                self.w[:, out == 1] = self.w[:, out == 1] + w_tmp[:, out == 1]

                self.w[self.w > self.weight_limit] = self.weight_limit
                self.w[self.w < -self.weight_limit] = -self.weight_limit
        outputs = np.stack(outputs)
        return outputs
