from CartpoleSNN import evolve_SNN_on_cartpole
from SS_IO import load_net_parameters

from SSNetVisualizator import SSNetVisualizer
from SimpleSNN import SpikeNN

snn = SpikeNN.load('winner_model')
# snn = SpikeNN.load('models/4_neurons_model')
net_params = load_net_parameters('net-config.json')
logs, score = evolve_SNN_on_cartpole(snn, net_params['Encoding'], render=False, logging=True, eps=1)
snn_outs, inputs_logs, neurons_logs, outputs_logs, observations, p_observations = logs
print(score)
print(neurons_logs[0][0].shape)
step = 6

for k, obs in enumerate(observations[0]):
    print(observations[0][k], '=>', p_observations[0][k])


print(neurons_logs[0][step][0, 8:])
print(type(neurons_logs[0][1]), neurons_logs[0][1].shape)

# net_visualizator = SSNetVisualizer(snn, neurons_logs[0][step][0], outputs_logs[0][step][0], neurons_labels=[0, 1, 2])
# net_visualizator.run()
