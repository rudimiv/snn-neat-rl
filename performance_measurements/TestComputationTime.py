#!/home/rudimiv/miniconda3/bin/python

import torch
import timeit
import time
import  numpy as np
import matplotlib.pyplot as plt


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


# sparse matrix, binary multiplication
def test(sizes, repeat, generator, f1, f2, cuda):
    if isinstance(repeat, int):
        repeat_arr = np.full((len(sizes)), repeat)
    else:
        repeat_arr = np.array(repeat)

    sizes = np.array(sizes)
    results_f1 = []
    results_f2 = []

    print(f1, f2)
    for size, r in zip(sizes, repeat_arr):
        data = generator(size, cuda)
        results_f1.append(perform(data, r, f1, cuda))
        results_f2.append(perform(data, r, f2, cuda))
        print(size, results_f1[-1], results_f2[-1])



    return sizes, results_f1, results_f2


def perform(data, rep, func, cuda):
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(rep):
        func(*data)

    if cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / rep

def generate_data_for_mul(size, cuda):
    weights = torch.rand((size, size))
    vector = torch.rand((size, 1)) > 0.5
    if cuda:
        device = torch.device('cuda:0')
        weights = weights.to(device)
        vector = vector.to(device)

    return weights, vector


def generate_data_for_heaviside(size, cuda):
    vector = torch.rand((size, 1))
    if cuda:
        device = torch.device('cuda:0')
        vector = vector.to(device)

    return (vector,)

def classical_multiply(weights, spikes):
    '''
    :param weights: N * N tensor
    :param spikes: N * 1
    :return:
    '''
    return weights.matmul(spikes.float())

def masked_multiply(weights, spikes):
    '''
    :param weights: N * N tensor
    :param spikes:  N * 1
    :return:
    '''

    return weights.masked_fill(~spikes.t(), 0.0).sum(dim=1)
    # return weights.sum(dim=1)
    # return spikes.masked_fill(~spikes.t(), 0.0)

def sparse_classical_multipy(weights, spikes):
    pass

def sparse_masked_multiply(weights, spikes):
    pass


def masked_selection(vector):
    output_t = vector >= 0.5
    return vector.masked_fill_(output_t, 5.0)

def heaviside_selection(vector):
    return 5.0 * torch.heaviside(vector - 0.5, torch.zeros_like(vector))

def test_mul(size_arr, repeats, cpu, gpu):
    data = []
    names = []

    if cpu:
        size_arr_, classical, masked = test(size_arr, repeats, generate_data_for_mul,
                                            classical_multiply, masked_multiply, False)
        data.append(classical)
        data.append(masked)
        names.append('classical')
        names.append('masked')

    if gpu:
        size_arr_, cuda_classical, cuda_masked = test(size_arr, repeats,
                                            generate_data_for_mul, classical_multiply,
                                                      masked_multiply, True)
        data.append(cuda_classical)
        data.append(cuda_masked)
        names.append('cuda_classical')
        names.append('cuda_masked')

    return size_arr_, data, names


def test_heaviside(size_arr, repeats, cpu, gpu):
    data = []
    names = []

    if cpu:
        size_arr_, heaviside, masked = test(size_arr, repeats, generate_data_for_heaviside,
                                            heaviside_selection, masked_selection, False)
        data.append(heaviside)
        data.append(masked)
        names.append('heaviside')
        names.append('masked')

    if gpu:
        size_arr_, cuda_heaviside, cuda_masked = test(size_arr, repeats,
                                            generate_data_for_heaviside, heaviside_selection,
                                                      masked_selection, True)
        data.append(cuda_heaviside)
        data.append(cuda_masked)
        names.append('cuda_heaviside')
        names.append('cuda_masked')

    return size_arr_, data, names

if __name__ == '__main__':
    size_arr = [10, 20, 50, 100, 150, 200, 400, 600, 800, 1000, 5000, 10000, 20000, 30000, 40000]
    # repeats =  [40, 40, 40,  40,  40,  40,  10,  10,  10,    5,    5,     5,     3,     2,     2]
    # repeats =  [40, 40, 40,  40,  40,  40,  10,  10,  10,    10,    10,     10,     5,     5,     5]

    # size_arr = [i for i in range(10 ** 4, 10 ** 5, 10000)]
    repeats = [10 for i in range(len(size_arr))]
    limit = -2
    cpu=True
    gpu=True
    # size_arr_, data, names = test_mul(size_arr[:limit], repeats[:limit], cpu, gpu)
    size_arr_, data, names = test_heaviside(size_arr[:limit], repeats[:limit], cpu, gpu)
    draw(size_arr_, data, names)
