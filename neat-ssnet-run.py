import argparse
import os

import neat
import torch
from neat.attributes import FloatAttribute, BoolAttribute
from neat.genes import BaseGene, DefaultConnectionGene
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output

from cartpole.CartpoleSNN import evolve_SNN_on_cartpole
from bipedal.BipedalSNN import evolve_SNN_on_bipedal
from SS_IO import load_net_parameters
import multiprocessing
import numpy as np
import networkx as nx
from SimpleSNN import SpikeNN


class SSNodeGene(BaseGene):
    _gene_attributes = []

    def distance(self, other, config):
        return 0.0


class SSConnection(BaseGene):
    _gene_attributes = [
        FloatAttribute('weight_m'), # add excitatory or inhibiitory
        BoolAttribute('enabled')
    ]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight_m - other.weight_m)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient


class SSGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        # values from config file are passed in param_dict
        param_dict['node_gene_type'] = SSNodeGene
        # param_dict['connection_gene_type'] = SSConnection
        param_dict['connection_gene_type'] = DefaultConnectionGene

        return DefaultGenomeConfig(param_dict)

def create_SSNet(genome, config, neuron_params):
    '''
    Receives a SSgenome and returns its phenotype
    :param genome:
    :param config:
    :return:
    '''
    genome_config = config.genome_config

    # Collect the nodes whose state is required to compute the final network output(s)
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

    # form matrix
    # Gather expressed connections.
    # connections = [cg.key for cg in genome.connections.values() if cg.enabled]
    node_inputs = {}
    G = nx.DiGraph()

    G.add_nodes_from(genome_config.input_keys)
    G.add_nodes_from(genome_config.output_keys)

    for cg in genome.connections.values():
        if not cg.enabled:
            continue

        i, o = cg.key

        if o not in required and i not in required:
            continue

        G.add_edge(i, o, weight=cg.weight)


    '''return SpikeNN(G, genome_config.input_keys, genome_config.output_keys, dt=config.dt, threshold=1.2, tau=0.3, rest=-0.07,
                   reset=-0.1, reversal=2.0, graded_input_potentials=True)'''

    return SpikeNN(G, genome_config.input_keys, genome_config.output_keys, **neuron_params)

class Experiment:
    def __init__(self, neat_config_path, net_config_path, evaluation_function, winner_file, cuda=False, logging=False):
        self.neat_config_path = neat_config_path
        self.net_config_path = net_config_path
        self.evaluation_function = evaluation_function
        self.winner_file = winner_file
        self.cuda = cuda
        self.logging = logging

        print('NEAT config path', self.neat_config_path)
        print('NET config path', self.net_config_path)
        # load configs
        self.config = neat.Config(SSGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, self.neat_config_path)
        self.net_params = load_net_parameters(self.net_config_path)

        # print net parameters
        print('Neuron parameters', self.net_params['LIF'])
        print('Encoding parameters', self.net_params['Encoding'])
        # Maybe it is better to separate Neuron and encoding parameters in two files
        self.neuron_parameters = self.net_params['LIF']
        self.encoding_parameters = self.net_params['Encoding']

    def simulate(self, genome, config):
        net = create_SSNet(genome, config, self.neuron_parameters)
        device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        # print('here:', device, self.logging)

        if self.logging:
            logs, scores = self.evaluation_function(
                net, self.encoding_parameters, eps=2, logging=True, device=device
            )
            snn_outs, inputs_logs, neurons_logs, outputs_logs, observations, p_observations = logs
        else:
            scores = self.evaluation_function(
                net, self.encoding_parameters, eps=2, logging=False, device=device
            )

        scores = min(scores), np.mean(scores), max(scores)
        print('{0:3d} nodes,  {1:3d} connections, total-score: {2:s}'.format(
            len(genome.nodes), len(genome.connections), str(scores))
        )
        return scores[0]

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.simulate(genome, config)

    def save_model(self, genome, config, filename):
        net = create_SSNet(genome, config, self.neuron_parameters)
        net.save(filename)

    def run(self):
        if self.encoding_parameters.get('plus_minus_separation') == True:
            # from neat/genome.py
            self.config.genome_config.num_inputs *= 2
            self.config.genome_config.input_keys = [-i - 1 for i in range(self.config.genome_config.num_inputs)]

        # config.output_nodes *= 2
        print(self.config.genome_config.num_inputs)
        pop = neat.population.Population(self.config)

        # Add a stdout reporter to show progress in the terminal.
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.simulate)
        # pe = neat.ParallelEvaluator(1, self.simulate)
        winner = pop.run(pe.evaluate, 3000)
        print(winner)
        self.save_model(winner, self.config, self.winner_file)
        # winner = pop.run(eval_genomes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conduct SSNet experiments with Gym environments')
    parser.add_argument('--net_config', required=True)
    parser.add_argument('--neat_config', required=True)
    parser.add_argument('--gym_env', required=True, choices=['cartpole','bipedal'])
    parser.add_argument('--winner_file', default='winner_model')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--verbose')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    print(local_dir)
    args.net_config = os.path.join(local_dir, args.net_config)
    args.neat_config = os.path.join(local_dir, args.neat_config)
    args.winner_file = os.path.join(local_dir, args.winner_file)

    print(args)

    if args.gym_env == 'cartpole':
        evaluation_function = evolve_SNN_on_cartpole
    elif args.gym_env == 'bipedal':
        evaluation_function = evolve_SNN_on_bipedal

    exp = Experiment(args.neat_config, args.net_config, evaluation_function,
                     args.winner_file, cuda=args.gpu)
    exp.run()

