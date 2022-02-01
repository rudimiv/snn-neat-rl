import numpy as np
import torch

class DataGenerator:
    # for DVSGesture and SpeechDataset see norse code. There are implementations
    @staticmethod
    def load_from_file(self):
        pass

    @staticmethod
    def bernoulli(batch_size, neurons, time, parameters):
        '''
        Probability based
        :param batch_size: number
        :param neurons: number
        :param time: number
        :param parameters: number or array
        :return: (batch_size, neurons, time) array of 0 and 1
        '''
        # isinstance(a, (int, long..)
        if np.isscalar(parameters):
            p = np.full((neurons,), parameters)
        else:
            p = parameters

        return (np.random.rand(batch_size, neurons, time) < p.reshape(1, p.shape[0], 1)).astype('int')

    @staticmethod
    def frequency(batch_size, neurons, time, rates):
        res = np.zeros((batch_size, neurons, time), dtype=np.int32)

        if np.isscalar(rates):
            # r = np.full((neurons,), rates)
            res[:,:,::rates] = 1
        else:
            res = np.zeros((batch_size, neurons, time), dtype=np.int32)

            for i, r in enumerate(rates):
                res[:, i, ::r] = 1

        return res

class Decoder:
    @staticmethod
    def count_frequencies(data):
        '''
        :param data (batch_size, neurons, time) array
        :return: (batch_size, neurons,) array
        '''
        return np.average(data, axis=2)

    @staticmethod
    def count_spikes(data):
        '''
        :param data (batch_size, neurons, time) array
        :return: (batch_size, neurons,) array
        '''
        return torch.sum(data, dim=2)

def bernoulli_test_SNN():
    pass

def bernoulli_test():
    data = DataGenerator.bernoulli(2, 3, 1000, 0.3)
    assert np.abs(np.average(data) - 0.3) < 0.01, f'Bernoulli estimation problems'

    data = DataGenerator.bernoulli(1, 3, 10000, np.array([0.3, 0.2, 0.4]))
    assert np.abs(np.average(data[0, 0,:]) - 0.3) < 0.01, f'Bernoulli estimation problems'
    assert np.abs(np.average(data[0, 2, :]) - 0.4) < 0.01, f'Bernoulli estimation problems'

def frequency_test():
    data = DataGenerator.frequency(2, 3, 2999, 10)
    assert np.abs(np.average(data) - 0.1) < 0.01, f'Frequncy generator problems'

    data = DataGenerator.frequency(2, 3, 2999, [10, 100, 2])
    assert np.abs(np.average(data[0, 0, :]) - 0.1) < 0.01, f'Frequncy generator problems'
    assert np.abs(np.average(data[0, 2, :]) - 0.5) < 0.01, f'Frequncy generator problems'