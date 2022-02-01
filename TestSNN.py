import numpy as np
import networkx as nx
import torch
from SimpleSNN import SpikeNN
from Visualizator import visualize_spike_tensor, visualize_neuron
from SpikeData import DataGenerator

import os
import unittest

class TestSNN(unittest.TestCase):
    def setUp(self):
        self.gr = nx.DiGraph()
        self.gr.add_nodes_from(range(0, 6))
        self.gr.add_edges_from([(1, 0), (2, 1), (4, 1), (3, 1), (4, 2), (4, 3), (5, 3)])

    def test_SNN_permutation(self):
        snn = SpikeNN(self.gr, [5,4], [0], tau=1.0, threshold=1.2, dt=0.1, rest=-0.2)
        self.assertEqual(type(snn), SpikeNN)

    def test_SNN_cuda(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
            snn = SpikeNN(self.gr, [5, 4], [0], tau=1.0, threshold=1.2, dt=0.1, rest=-0.2)
            # self.assertEqual(type(snn), SpikeNN)
            snn = snn.to(device)
            data = DataGenerator.bernoulli(1, 1, 110, 0.2)
            data = torch.from_numpy(data).to(device)
            result = snn.forward(data)
            print(result.device)

        else:
            print('CUDA is unavailable')

    def test_simple_SNN(self, visualize=False):
        gr = nx.DiGraph()
        gr.add_nodes_from(range(0, 6))
        gr.add_edges_from([(0, 1), (1, 2), (1, 4), (1, 3), (2, 4), (3, 4), (3, 5)])

        snn = SpikeNN(gr, [0], [5,4], tau=1.0, threshold=1.2, dt=0.1, rest=-0.2)

        common_neurons_log = []
        common_outputs_log = []
        common_inputs_log = []
        common_results = []

        for i in range(10):
            # data = np.random.randint(0, 2, (1, 100))
            # result, inputs_log, neurons_log, outputs_log = snn.forward(torch.from_numpy(data[np.newaxis, :, :]), logging=True)
            data = DataGenerator.bernoulli(1, 1, 110, 0.2)
            result, inputs_log, neurons_log, outputs_log = snn.forward(torch.from_numpy(data),logging=True)

            common_inputs_log.append(inputs_log)
            common_neurons_log.append(neurons_log)
            common_outputs_log.append(outputs_log)
            common_results.append(result)

        common_results = torch.cat(common_results, dim=2)
        common_inputs_log = torch.cat(common_inputs_log, dim=2)
        common_outputs_log = torch.cat(common_outputs_log, dim=2)
        common_neurons_log = torch.cat(common_neurons_log, dim=2)

        self.assertEqual(common_results.shape, (1, 2, 1100))
        self.assertEqual(common_inputs_log.shape, (1, 6, 1100))
        self.assertEqual(common_outputs_log.shape, (1, 6, 1100))
        self.assertEqual(common_neurons_log.shape, (1, 6, 1100))


        if visualize:
            params = snn.get_params()
            neuron_to_visualize = 1
            visualize_neuron(common_inputs_log[0, neuron_to_visualize, :100],
                             common_neurons_log[0, neuron_to_visualize, :100],
                             common_outputs_log[0, neuron_to_visualize, :100],
                             params)

    def test_snn_serialization(self):
        gr = nx.DiGraph()
        gr.add_nodes_from(range(0, 6))
        gr.add_edges_from([(1, 0), (2, 1), (4, 1), (3, 1), (4, 2), (4, 3), (5, 3)])

        snn = SpikeNN(gr, [5, 4], [0], tau=1.0, threshold=1.2, dt=0.1, rest=-0.2)

        snn.save('test_save')
        snn_l = SpikeNN.load('test_save')
        # print(snn.G.nodes)
        # print(snn.G.edges)
        os.remove('test_save')
        # self.assertEqual(snn, snn_l)
        self.assertEqual(set(snn.G.nodes), set(snn_l.G.nodes))
        self.assertEqual(set(snn.G.edges), set(snn.G.edges))



if __name__ == '__main__':
    # data = torch.eye(10, 1000, dtype=torch.uint8)
    # simple_test_SNN()
    # check_SNN_permutation()
    # check_snn_serialization()
    unittest.main()
