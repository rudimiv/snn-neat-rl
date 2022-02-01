import numpy as np
import networkx as nx
import torch
from SimpleSNN import SpikeNN
from Visualizator import visualize_spike_tensor, visualize_neuron
from SpikeData import DataGenerator
from SSNetVisualizator import SSNetVisualizer

import os
import unittest


def bernouilli_test_SNN():
    gr = nx.DiGraph()
    gr.add_nodes_from(range(0, 6))
    gr.add_edges_from([(0, 1), (1, 2), (1, 4), (1, 3), (2, 4), (3, 4), (3, 5)])

    matrix = nx.adjacency_matrix(gr)
    print(matrix)

    snn = SpikeNN(gr, [0], [4, 5], tau=0.3, threshold=0.8, dt=1)
    data = np.random.randint(0, 2, (1, 100))
    result = snn.forward(torch.from_numpy(data[np.newaxis, :, :]))
    visualize_spike_tensor(result[0][0].numpy())
    # visualize_spike_tensor(data)
    print(result[0].shape, data.shape)


def test_graded_potentials():
    # create the base of defaults graphs
    gr = nx.DiGraph()
    gr.add_nodes_from(range(0, 6))
    gr.add_edges_from([(0, 1), (0,2 ) (1, 4), (1, 3), (2, 4), (3, 4), (3, 5)])
    matrix = nx.adjacency_matrix(gr)
    print(matrix)

    snn = SpikeNN(gr, [0], [4, 5], tau=0.3, threshold=0.8, rest=-0.07, reset=-0.1, dt=0.1,
                  reversal=2.0, graded_input_potentials=True)

    b, a = 2.0, 0.0
    # data = (b - a) * np.random.random((1, 1)) + a
    data = np.array([[0.9]])
    # data = np.repeat(data[np.newaxis, :], 100, axis=2)
    print(data.shape)
    result, inputs_log, neurons_log, outputs_log = snn.forward(torch.from_numpy(data).float(), logging=True, time_=100)
    print(result.shape)
    print(neurons_log.shape)

    net_visualizator = SSNetVisualizer(snn, neurons_log[0], outputs_log[0], neurons_labels=[0, 1, 2],
                                       edge_animation=False, node_animation=True)
    net_visualizator.run()

if __name__ == '__main__':
    test_graded_potentials()