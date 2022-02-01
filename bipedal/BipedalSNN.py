import gym
import torch
import copy
from SimpleSNN import SpikeNN
import numpy as np
import networkx as nx
from SpikeData import DataGenerator, Decoder

def preprocess_data(observation, encoding, env):
    # print(encoding)
    algorithm = encoding['algorithm']
    plus_minus_separation = encoding.get('plus_minus_separation', False)

    # data separating
    if algorithm == 'clip' and plus_minus_separation:
        res_p = np.zeros((observation.shape[0] * 2,))

        mins = env.observation_space.low
        maxs = env.observation_space.high
        # https://github.com/openai/gym/wiki/BipedalWalker-v2

        mins[0], maxs[0] = 0, 2 * np.pi
        mins[2], maxs[2] = -1, 1
        mins[3], maxs[3] = -1, 1
        mins[8], maxs[8] = 0, 1
        mins[13], maxs[13] = 0, 1

        mins.fill(-5)
        maxs.fill(5)
        # if inf => -10


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
        mins[0], maxs[0] = 0, 2 * np.pi
        mins[2], maxs[2] = -1, 1
        mins[3], maxs[3] = -1, 1
        mins[8], maxs[8] = 0, 1
        mins[13], maxs[13] = 0, 1

        mins[np.isinf(mins)] = -5
        maxs[np.isinf(maxs)] = 5
        # code environment data
        p = np.clip((observation - mins) / (maxs - mins), 0, 1)
    else:
        raise ValueError('Invalid encoding scheme')

    return p

import time
def evolve_SNN_on_bipedal(net, encoding, eps=1, time_one_op=50,
                          render=False, logging=False, max_steps=None, device='cpu'):

    start = time.time()
    env = gym.make('BipedalWalker-v3')
    # print('time:', time.time() - start)
    # print('loggin in evolve:', logging)
    # cuda = next(net.parameters()).is_cuda
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
        final_reward = 0.0

        for i in range(max_steps):
            if render:
                env.render()


            start = time.perf_counter()
            p = preprocess_data(observation, encoding, env)
            # print('preprocess:', time.perf_counter() - start)
            start = time.perf_counter()
            if net.graded_input_potentials:
                snn_in = torch.tensor(p[np.newaxis, :], dtype=torch.float, device=device)

                snn_out, inputs_log, neurons_log, outputs_log = snn.forward(
                    snn_in, logging=logging, time_=time_one_op
                )
            else:
                snn_in = torch.from_numpy(DataGenerator.bernoulli(1, 4, time_one_op, p)).to(device)
                snn_out, inputs_log, neurons_log, outputs_log = snn.forward(snn_in, logging=logging)

            # print('forward:', time.perf_counter() - start)
            if logging:
                snn_outs[ep].append(snn_out.to('cpu'))
                print(inputs_log.shape, inputs_log.device)
                inputs_logs[ep].append(inputs_log)
                neurons_logs[ep].append(neurons_log)
                outputs_logs[ep].append(outputs_log)
                observations[ep].append(observation)
                preprocessed_observations[ep].append(p)

            # print('Decoder', Decoder.count_spikes(snn_out))
            out_rates = Decoder.count_spikes(snn_out).float() / time_one_op
            actions = out_rates[0, :4] - out_rates[0, 4:]
            actions = actions.to('cpu')
            # (actions)
            # print(actions)
            observation, r, done, info = env.step(actions.numpy())
            total_steps += 1
            final_reward += r
            # print(r, final_reward)
            if done:
                break

        scores.append(final_reward)

    if logging:
        return (snn_outs, inputs_logs, neurons_logs, outputs_logs, \
               observations, preprocessed_observations), scores
    else:
        return scores


def test_gym():
    env = gym.make('BipedalWalker-v3')

    print(env.observation_space)
    print(env.action_space)

    for ep in range(1):
        obs = env.reset()
        final_reward = 0.0
        for i in range(100):
            print(obs)
            env.render()
            action = env.action_space.sample()
            print(action)
            obs, r, done, _ = env.step(action)
            final_reward += r
            print('reward', r, final_reward)
            if done:
                print(i)
                break

if __name__ == '__main__':
    test_gym()