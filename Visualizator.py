import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_spike_tensor(input, stepsize=10):
    '''
    :param input: [neurons, time]
    :return:
    '''
    neurons, time = input.shape

    fig, ax = plt.subplots(1,1, figsize=(time, neurons), dpi=10)
    ax.pcolor(input, cmap='binary', vmax=1, vmin=0)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))

    plt.show()

def visualize_neuron(neuron_input, neuron_potential, neuron_output, params=None, neuro_raster=True, stepsize=5):
    '''
    :param neuron_potentials: (time,)
    :param neuron_inputs: (time,)
    :param neuron_outputs: (time,)
    :param params: tuple (tau, dt, u_rest, u_reset, threshold)
    :return: nothing
    '''
    time = neuron_potential.shape[0]
    time_arr = np.arange(time)

    if torch.is_tensor(neuron_potential):
        neuron_potential = neuron_potential.numpy()

    if torch.is_tensor(neuron_input):
        neuron_input = neuron_input.numpy()

    if torch.is_tensor(neuron_output):
        neuron_output = neuron_output.numpy()

    delta_lim = 5

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12, 7))

    ax1.step(range(time), neuron_input)
    # ax1.set_title('Neuron input')
    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel('Neuron input')
    start, end = ax1.get_xlim()
    ax1.set_xlim(-delta_lim, time + delta_lim)
    ax1.xaxis.set_ticks(np.arange(0, time, stepsize))

    # ax4.plot(neuron_input)
    # ax4.set_title('Neuron input')
    # ax4.grid(True)
    # ax4.set_xlabel('t')
    # ax4.set_ylabel('Neuron input')

    ax2.plot(neuron_potential, '.-',color='tomato')
    # ax2.set_title('Neuron potential')
    ax2.grid(True)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Neuron potential')
    ax2.set_xlim(-delta_lim, time + delta_lim)
    ax2.xaxis.set_ticks(np.arange(0, time, stepsize))

    if params is not None:
        tau, dt, rest_p, reset_p, threshold = params

        ax2.plot(time_arr, np.full_like(time_arr, threshold, dtype=np.float32), ls='--', alpha =0.4)
        ax2.plot(time_arr, np.full_like(time_arr, rest_p, dtype=np.float32), ls='--', alpha =0.4)
        # ax2.plot(range(time), u_reset)

    if neuro_raster:
        neuron_output = np.where(neuron_output > 0.5)[0]
        ax3.eventplot(neuron_output)
    else:
        # may be 1 shift is needed
        ax3.step(range(time), neuron_output)
        # ax3.set_title('Neuron output')

    ax3.grid(True)
    ax3.set_xlabel('t')
    ax3.set_ylabel('Neuron output')
    ax3.set_xlim(-delta_lim, time + delta_lim)
    ax3.xaxis.set_ticks(np.arange(0, time, stepsize))

    plt.show()

def test_visualizator():
    data = np.random.randint(0, 2, (10, 100))
    visualize_spike_tensor(data)