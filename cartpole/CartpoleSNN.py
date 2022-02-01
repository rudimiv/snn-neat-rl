import gym
import torch
import copy
from SimpleSNN import SpikeNN
import numpy as np
import networkx as nx
from SpikeData import DataGenerator, Decoder

def create_Simple_SNN():
    gr = nx.DiGraph()
    gr.add_nodes_from(range(0, 8))

    edges = []

    for i in [0, 1, 2, 3]:
        for j in [4, 5, 6]:
            edges.append((i,j))

    for i in [4]:
        for j in [7, 8]:
            edges.append((i,j))

    edges.append((6,8))
    edges.append((5, 8))

    gr.add_edges_from(edges)

    # matrix = nx.adjacency_matrix(gr)
    # print(matrix)

    snn = SpikeNN(gr, [0, 1, 2, 3], [7, 8], tau=0.3, threshold=1.5, dt=0.1)

    return snn


def test_snn_cartpole(eps=1, time_one_op=100):
    env = gym.make('CartPole-v1')
    mins = env.observation_space.low
    maxs = env.observation_space.high

    mins[[1,3]] = -10
    maxs[[1,3]] = 10

    # print(mins, maxs)

    for ep in range(eps):
        observation = env.reset()
        snn = create_Simple_SNN()

        for i in range(100):
            env.render()
            # code environment data
            p = np.clip((observation - mins) / (maxs - mins), 0, 1)

            snn_in = torch.from_numpy(DataGenerator.bernoulli(1, 4, time_one_op, p))

            # snn_out = snn.forward(snn_in)
            snn_out, inputs_log, neurons_log, outputs_log = snn.forward(snn_in, logging=True)
            # print(snn_in[0].t())
            # print(outputs_log[0].t())

            # visualize_neuron(inputs_log[0, 7, :100], neurons_log[0, 7, :100], outputs_log[0, 7, :100], snn.get_params())

            # visualize_neuron(inputs_log[0, 8, :100],neurons_log[0, 8, :100],outputs_log[0, 8, :100],snn.get_params())

            print(Decoder.count_spikes(snn_out))
            action = int(torch.argmax(Decoder.count_spikes(snn_out))[0])
            # print(action)

            observation, r, done, info = env.step(action)

            if done:
                print(i)
                break


def preprocess_data(observation, encoding, env):
    # print(encoding)
    algorithm = encoding['algorithm']
    plus_minus_separation = encoding.get('plus_minus_separation', False)

    # data separating
    if algorithm == 'clip' and plus_minus_separation:
        res_p = np.zeros((observation.shape[0] * 2,))

        mins = env.observation_space.low
        maxs = env.observation_space.high

        mins[[1, 3]] = -10
        maxs[[1, 3]] = 10

        clipped = np.clip((2 * observation) / (maxs - mins), -1, 1)

        for i, val in enumerate(clipped):
            if val >= 0:
                res_p[i * 2] = val
            else:
                res_p[i * 2 + 1] = -val


        p = res_p + 0.5
        # print(p)

    elif algorithm == 'gauss':
        gauss_std = encoding.get('gauss_std')
        p = np.exp(-((observation / gauss_std) ** 2) / 2) / (gauss_std * np.sqrt(2 * np.pi))
    elif algorithm == 'clip' and not plus_minus_separation:
        mins = env.observation_space.low
        maxs = env.observation_space.high

        # print(mins, file=sys.stderr)
        mins[[1, 3]] = -10
        maxs[[1, 3]] = 10
        # code environment data
        p = np.clip((observation - mins) / (maxs - mins), 0, 1)
    else:
        raise ValueError('Invalid encoding scheme')



    return p

def evolve_SNN_on_cartpole(net, encoding, eps=1, time_one_op=100,
                           render=False, logging=False, max_steps=None, device='cpu'):
    env = gym.make('CartPole-v1')

    # print(mins, maxs)
    snn_outs = []
    inputs_logs = []
    neurons_logs = []
    outputs_logs = []
    scores = []
    observations = []
    preprocessed_observations = []

    for ep in range(eps):
        observation = env.reset()
        snn = copy.deepcopy(net)
        snn = snn.to(device)

        max_steps = max_steps if max_steps else env._max_episode_steps + 1

        if logging:
            snn_outs.append([])
            inputs_logs.append([])
            neurons_logs.append([])
            outputs_logs.append([])
            observations.append([])
            preprocessed_observations.append([])

        total_steps = 0

        for i in range(max_steps):
            if render:
                env.render()

            p = preprocess_data(observation, encoding, env)

            if net.graded_input_potentials:
                snn_in = torch.tensor(p[np.newaxis, :], dtype=torch.float)
                snn_out, inputs_log, neurons_log, outputs_log = snn.forward(
                    snn_in, logging=True, time_=time_one_op
                )
            else:
                snn_in = torch.from_numpy(DataGenerator.bernoulli(1, 4, time_one_op, p)).to(device)
                snn_out, inputs_log, neurons_log, outputs_log = snn.forward(snn_in, logging=True)

            if logging:
                snn_outs[ep].append(snn_out)
                inputs_logs[ep].append(inputs_log)
                neurons_logs[ep].append(neurons_log)
                outputs_logs[ep].append(outputs_log)
                observations[ep].append(observation)
                preprocessed_observations[ep].append(p)

            action = int(torch.argmax(Decoder.count_spikes(snn_out)).item())
            observation, r, done, info = env.step(action)
            total_steps += 1
            if done:
                break

        scores.append(total_steps)

    if logging:
        return (snn_outs, inputs_logs, neurons_logs, outputs_logs, \
               observations, preprocessed_observations), scores
    else:
        return scores

def test_gym():
    env = gym.make('CartPole-v0')

    print(env.observation_space)
    print(env.action_space)

    for ep in range(10):
        obs = env.reset()
        for i in range(100):
            print(obs)
            env.render()
            action = env.action_space.sample()
            print(action)
            obs, r, done, _ = env.step(action)

            if done:
                print(i)
                break

if __name__ == '__main__':
    test_snn_cartpole()
    # test_gym()