#!/home/rudimiv/miniconda3/bin/python

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from SimpleSNN import SpikeNN


def draw(size_arr, results, names):
    fig, ax = plt.subplots(1)

    for i in range(len(results)):
        ax.plot(size_arr, results[i], label=names[i])

    ax.set_xlabel('size')
    ax.set_ylabel('time')
    ax.set_xscale('log')
    ax.grid(True)

    ax.legend()
    plt.show()


def generate_net_n(neurons, source_nodes, output_nodes, graded_potentials=True):
    # add connections density
    graph = nx.erdos_renyi_graph(neurons, 0.5, directed=True)
    # di_graph = nx.DiGraph(graph.nodes)
    # for (i, v) in graph.edges():


    input_n = list(graph.nodes)[:source_nodes]
    output_n = list(graph.nodes)[output_nodes:]

    return SpikeNN(graph, input_n, output_n, dt=0.1, threshold=1.2, reset=-0.1,
                   rest=-0.07, tau=0.3, reversal=2.0, graded_input_potentials=graded_potentials)

def generate_input_vector(size, graded_potentials, batch_size=1):
    if graded_potentials:
        return torch.rand((batch_size, size))
    else:
        pass



def perform(net, vector, rep, cuda):
    if cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(rep):
        # print(vector.shape)
        net.forward(vector, time_=100)

    if cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / rep

def test_cpu_gpu(size_arr, repeats, nets, vectors):
    results_cpu = []
    results_gpu = []
    cuda = True

    # print(nets, vectors)

    for i in range(len(size_arr)):
        nets[i].reset()
        print('evaluate size = ', size_arr[i])
        results_cpu.append(perform(nets[i], vectors[i], repeats[i], False))
        # results_cpu.append(1.0)
        if cuda:
            nets[i].reset()
            results_gpu.append(perform(nets[i].to('cuda:0'), vectors[i].to('cuda:0'), repeats[i], True))
        else:
            results_gpu.append(0)

    return size_arr, results_cpu, results_gpu

def test_cpu_gpu_net(size_arr, repeats, batches=[1]):
    data = []
    names = []

    nets = []
    for size in size_arr:
        print('generate size = ', size)
        input_n = int(size * 0.1)
        output_n = int(size * 0.1)
        nets.append(generate_net_n(size, input_n, output_n))


    for bs in batches:
        vectors = []
        for size in size_arr:
            input_n = int(size * 0.1)
            vectors.append(generate_input_vector(input_n, True, bs))

        size_arr_, cpu_net, cuda_net = test_cpu_gpu(size_arr, repeats, nets, vectors)
        data.append(cuda_net)
        data.append(cpu_net)
        names.append(f'cuda_net bs={bs}')
        names.append(f'cpu_net bs={bs}')

    return size_arr_, data, names

if __name__ == '__main__':
    size_arr = [10, 20, 50, 100, 150, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 12000]
    # repeats =  [40, 40, 40,  40,  40,  40,  10,  10,  10,    5,    5,     5,     3,     2,     2]
    # repeats =  [40, 40, 40,  40,  40,  40,  10,  10,  10,    10,    10,     10,     5,     5,     5]

    # size_arr = [i for i in range(10 ** 4, 10 ** 5, 10000)]
    repeats = []
    for size in size_arr:
        if size <= 1000:
            repeats.append(10)
        elif size > 1000 and size <= 4000:
            repeats.append(3)
        else:
            repeats.append(1)

    # repeats = [10 for i in range(len(size_arr))]
    batches = [1, 2, 5, 10]
    limit = -3

    # size_arr_, data, names = test_mul(size_arr[:limit], repeats[:limit], cpu, gpu)
    size_arr_, data, names = test_cpu_gpu_net(size_arr[:limit], repeats[:limit], batches)
    draw(size_arr_, data, names)
