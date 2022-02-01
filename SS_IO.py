import json


def write_logs_to_file(filename, snn_out, inputs_log, neurons_log, outputs_log):
    '''
    :param snn_out: [output_neurons, time]
    :param inputs_log:  [total_neurons_number, time]
    :param neurons_log: [total_neurons_number, time]
    :param outputs_log: [total_neurons_number, time]
    :return:
    '''


def read_logs_from_file(filename):
    pass


def load_net_parameters(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data