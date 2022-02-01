import torch
from torch.nn.parameter import Parameter
import networkx as nx
import numpy as np
import math
import pickle


class SpikeNN(torch.nn.Module):
    def __init__(self, G, input_neurons_o, output_neurons_o, dt, tau, threshold,
                 reset=0.0, rest=0.0, reversal=0.0, refr_period=None, low_bound=None, graded_input_potentials=False):
        super().__init__()

        self.G = G
        self.input_neurons_o = input_neurons_o
        self.output_neurons_o = output_neurons_o

        # second parameter defines order of elements in matrix

        self.input_neurons = []
        self.output_neurons = []

        for i, node in enumerate(G.nodes):
            # add indexes
            if node in self.input_neurons_o:
                self.input_neurons.append(i)

            if node in self.output_neurons_o:
                self.output_neurons.append(i)

        # self.weights = torch.from_numpy(weights.todense()).float().t()
        self.weights_ = torch.from_numpy(nx.to_numpy_matrix(G, list(G.nodes))).float().t()
        self.number_of_neurons = self.weights_.shape[0]


        # it is possible to replace variables by arrays of these values
        # permutation
        self.tau = tau
        self.threshold = threshold
        self.reset_p = reset
        self.rest_p = rest
        self.reversal_p = reversal
        self.dt = dt
        self.koeff = np.exp(-self.dt / self.tau)

        if refr_period is None:
            self.refr_period = dt
        else:
            self.refr_period = refr_period

        self.low_bound = low_bound
        self.graded_input_potentials = graded_input_potentials
        self.neurons_log = None
        self.outputs_log = None

        # self.register_buffer

        # input_neurons don't overlap with output_neurons
        # it is possible to use a permutation matrix P PAP
        permutation = self.input_neurons + \
                      list(set(range(self.number_of_neurons)) - set(self.input_neurons) - set(self.output_neurons))+ \
                    self.output_neurons

        m = {n:i for i, n in enumerate(G.nodes)}
        self.G_relabeled = nx.relabel_nodes(self.G, m)
        m = {n:i for i, n in enumerate(permutation)}
        # print('m2:', m)
        self.G_relabeled = nx.relabel_nodes(self.G_relabeled, m)
        # print('after: ', self.G_relabeled.nodes)
        # print(permutation)
        # print(self.G_relabeled.edges)
        # permute: input_neurons - first K rows of matrix
        # permute: output_neurons - last M rows of matrix
        # print(permutation)
        # print('weights original:\n', self.weights)
        # torch.Tensor is an alias
        self.weights = Parameter(torch.FloatTensor(self.number_of_neurons, self.number_of_neurons),
                                 requires_grad=False)

        if self.graded_input_potentials:
            # u = (E_l + b * E_syn) / (1 + b) + 1/tau * W * o * e^-(t-t_i)...
            self.graded_weights = None

        self.reset()


    def reset(self, logging_reset=True):
        self.output_t = None
        self.neuron_states = None
        self.refr_count = None

        self.weights = Parameter(self.weights_)

        if logging_reset:
            self.neurons_log = None
            self.outputs_log = None
            self.inputs_log = None

    def get_params(self):
        return self.tau, self.dt, self.rest_p, self.reset_p, self.threshold

    def forward(self, input, refractor_period=True, logging=False, time_=None):
        '''
        :param input: tensor [batch, input_neurons, time] or [batch, input_neurons] for the
        case of the graded potentials.
        :return: if logging is False [batch, output_neurons, time]
        :return: if loggin is True  snn_out: [batch, output_neurons, time],
                                    inputs:  [batch, total_neurons_number, time],
                                    neurons: [batch, total_neurons_number, time],
                                    outputs: [batch, total_neurons_number, time]
        '''
        if len(input.shape) == 3:
            batch_size,_,time = input.shape
        elif len(input.shape) == 2:
            batch_size, _ = input.shape
            time = time_
        else:
            raise ValueError('dimension mismatch')

        device = self.weights.device
        # print(device)
        # neuron outputs (input neurons + intermediate neurons + output neurons_
        if self.output_t is None:
            self.output_t = torch.zeros(batch_size, self.number_of_neurons, 1, dtype=torch.uint8, device=device)

        # neuron potential
        if self.neuron_states is None:
            self.neuron_states = torch.zeros(batch_size, self.number_of_neurons, 1, dtype=torch.float32, device=device)

        if self.refr_count is None:
            self.refr_count = torch.zeros(batch_size, self.number_of_neurons, 1, dtype=torch.float32, device=device)

        if self.neurons_log is None and logging:
            self.neurons_log = torch.empty(batch_size, self.number_of_neurons, time, dtype=torch.float32)

        if self.outputs_log is None and logging:
            self.outputs_log = torch.empty(batch_size, self.number_of_neurons, time, dtype=torch.uint8)

        if self.inputs_log is None and logging:
            self.inputs_log = torch.empty(batch_size, self.number_of_neurons, time, dtype=torch.float32)

        if self.graded_input_potentials:
            # betas, Ks and hence koeffs are depend on inputs
            self.betas = self.weights[:, :len(self.input_neurons)].matmul(input[:, :, None])
            self.K = (self.betas * self.reversal_p + self.rest_p) / (1 + self.betas)
            self.graded_koeffs = torch.exp(-self.dt / self.tau * (1 + self.betas))


            '''print(self.reversal_p)
            print(self.weights[:, :len(self.input_neurons)])
            print('betas    ', self.betas[0].t())
            print('K        ', self.K[0].t())
            print('gr_coeffs', self.graded_koeffs[0].t())'''

        # snn out
        output = torch.zeros(batch_size, len(self.output_neurons), time, dtype=torch.uint8, device=device)

        for t in range(time):
            if not self.graded_input_potentials:
                self.output_t[:,:len(self.input_neurons), 0] = input[:, :, t]

            if logging:
                self.outputs_log[:, :, t] = self.output_t[:, :, 0]
                self.inputs_log[:, :, t] = self.weights.matmul(self.output_t.float())[:, :, 0]
                # self.neurons_log[:, :, t] = self.neuron_states[:, :, 0]

            if refractor_period:
                if self.graded_input_potentials:
                    self.neuron_states = (self.neuron_states - self.K) * self.graded_koeffs + self.K + \
                                         self.weights.matmul(self.output_t.float())
                else:
                    self.neuron_states = (self.neuron_states - self.rest_p) * self.koeff + self.rest_p + \
                                         self.weights.matmul(self.output_t.float())

                # print('neuron_states device:', self.neuron_states.device)

                self.neuron_states.masked_fill_(self.refr_count > 0, self.reset_p)

                if logging:
                    self.neurons_log[:, :, t] = self.neuron_states[:, :, 0]

                # ~ heaviside
                self.output_t = self.neuron_states >= self.threshold
                self.refr_count -= self.dt
                self.refr_count.masked_fill_(self.output_t, self.refr_period)
                # low boundaries
                if self.low_bound is not None:
                    self.neuron_states.masked_fill_(self.neuron_states < self.low_bound, self.low_bound)

            else:
                self.neuron_states = ((self.neuron_states - self.rest_p) * self.koeff)
                self.neuron_states.masked_fill_(self.output_t > 0, 0)
                self.neuron_states += self.rest_p + self.weights.matmul(self.output_t.float())

                if logging:
                    self.neurons_log[:, :, t] = self.neuron_states[:, :, 0]

                self.output_t = self.neuron_states >= self.threshold

            # extract output
            output[:, :, t] = self.output_t[:, -len(self.output_neurons):, 0]
            # print('output device:     ', output.device, logging)
            if logging:
                print('outputs_log device:', self.outputs_log.device)

        if logging:
            return output, self.inputs_log, self.neurons_log, self.outputs_log
        else:
            return output, None, None, None

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)




