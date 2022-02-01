import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt # ; plt.close('all')
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from SimpleSNN import SpikeNN
from SpikeData import DataGenerator
import math



class SSNetVisualizer:
    def __init__(self, net, neuron_potentials=None, neuron_outputs=None, frames=None, neurons_labels=[],
                        visualize_inputs=False, visualize_outputs=True, visulize_potential=True, pos=None,
                        node_animation=True, edge_animation=False):

        self.neuron_potentials = neuron_potentials
        self.neuron_outputs = neuron_outputs
        self.frames = frames
        self.total_len = self.neuron_potentials.shape[1]
        self.neuron_labels = neurons_labels
        self.net = net
        self.G = net.G_relabeled

        self.neuron_plot_koeff = sum([visualize_inputs, visualize_outputs, visulize_potential])
        self.visualize_inputs = visualize_inputs
        self.visualize_outputs = visualize_outputs
        self.visulize_potential = visulize_potential

        self.tacts = 100

        self.fig = plt.figure(figsize=(20, 10))
        neurons_plots = len(neurons_labels)

        # if 3 neurons with details 2 or  3 graphs
        # or 9 neurons one graph type

        if neurons_plots * self.neuron_plot_koeff >= 9:
            return


        gs = gridspec.GridSpec(nrows=(self.neuron_plot_koeff + 3), ncols = 3)

        self.ax_net = self.fig.add_subplot(gs[:2, :2])
        self.ax_gym = self.fig.add_subplot(gs[:2, 2])
        self.neuron_axes = []

        # if self.neuron_plot_koeff > 1:
        for i in range(neurons_plots):
            self.neuron_axes.append([])
            count = 0

            if self.visualize_inputs:
                self.neuron_axes[i].append(self.fig.add_subplot(gs[3 + count, i]))
                count += 1
            else:
                self.neuron_axes[i].append(None)

            if self.visulize_potential:
                self.neuron_axes[i].append(self.fig.add_subplot(gs[3 + count, i]))
                count += 1
            else:
                self.neuron_axes[i].append(None)

            if self.visualize_outputs:
                self.neuron_axes[i].append(self.fig.add_subplot(gs[3 + count, i]))
            else:
                self.neuron_axes[i].append(None)

        # define neuron positions
        if pos is None:
            # pos = nx.spring_layout(G)
            # pos = nx.kamada_kawai_layout(G)
            self.pos = nx.drawing.nx_agraph.graphviz_layout(self.G, prog='/Users/rudimiv/miniconda3/bin/dot', args="-Grankdir=LR")

        self.node_animation = node_animation
        self.edge_animation = edge_animation
        self._anim_is_running = False



    def onClick(self, event):
        if self._anim_is_running:
            self.animator.event_source.stop()
            self._anim_is_running = False
        else:
            self.animator.event_source.start()
            self._anim_is_running = True

    def _init(self):
        # init net visualization
        self.node_labels = {i: i for i in list(self.G.nodes)}
        eps = (self.net.threshold - self.net.reset_p) / 20
        self.min_neuron = self.net.reset_p - eps
        self.max_neuron = self.net.threshold + eps
        self.node_sizes = [400 if node < len(self.net.input_neurons) or
                                  node >= (self.net.number_of_neurons - len(self.net.output_neurons)) else 200 for node in self.G.nodes]

        self.v_graph_nodes = nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax_net, node_color=np.zeros(len(self.G.nodes)),
                         cmap='Reds', vmin=self.min_neuron, vmax=self.max_neuron, node_size=self.node_sizes)

        self.v_graph_labels = nx.draw_networkx_labels(self.G, self.pos, ax=self.ax_net, labels=self.node_labels, font_color='black')
        self.v_graph_edges = nx.draw_networkx_edges(self.G, self.pos, ax=self.ax_net, edge_color=np.zeros(len(self.G.edges)))
        edge_labels = {edge: f'{self.G.get_edge_data(*edge).get("weight", 1.0):.2f}' for edge in self.G.edges}

        self.v_graph_weights = nx.draw_networkx_edge_labels(self.G, self.pos, ax=self.ax_net, edge_labels=edge_labels)
        # print(self.v_graph_edges)

        self.lines = []
        self.event_plots = []
        self.delta_lim = 5
        self.time_arr = np.arange(self.total_len)
        # init spike trains visualizers
        for i, neuron in enumerate(self.neuron_labels):
            if self.neuron_axes[i][0]:
                ax = self.neuron_axes[i][0]

                ax.grid(True)
                ax.set_xlabel('t')
                ax.set_ylabel(f'N_{neuron} input')

            if self.neuron_axes[i][1]:
                ax = self.neuron_axes[i][1]

                ax.grid(True)
                ax.set_xlabel('t')
                ax.set_ylabel(f'N_{neuron} potential')
                ax.plot(self.time_arr, np.full_like(self.time_arr, self.net.threshold, dtype=np.float32), ls='--', alpha=0.4)
                ax.plot(self.time_arr, np.full_like(self.time_arr, self.net.rest_p, dtype=np.float32), ls='--', alpha=0.4)

                self.lines.append(ax.plot([], [])[0])

            if self.neuron_axes[i][2]:
                ax = self.neuron_axes[i][2]

                ax.grid(True)
                ax.set_xlabel('t')
                ax.set_ylabel(f'N_{neuron} output')

                self.event_plots.append(ax.eventplot([])[0])
                # lw = 2

    def _animate(self, frame):

        '''nx.draw_networkx(self.G, self.pos, ax = self.ax_net, node_color=node_colors, labels=self.node_labels, font_color='black',
                         cmap='Reds', vmin=self.min_neuron, vmax=self.max_neuron, node_size=self.node_sizes,
                         edge_color=edge_colors)'''

        # nodes are just markers returned by plt.scatter;
        # node color can hence be changed in the same way like marker colors
        if self.node_animation:
            node_colors = np.array([self.neuron_potentials[node][frame] for node in self.G.nodes])
            self.v_graph_nodes.set_array(node_colors)
        # print(self.v_graph_nodes, self.v_graph_edges)
        # self.v_graph_edges.set_array(edge_colors)
        if self.edge_animation:
            edge_colors = np.array([1.0 if self.neuron_outputs[o][frame] else 0.0 for o, i in self.G.edges])
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax_net, edge_color=edge_colors)


        # visualize spike trains
        part = int(self.tacts  * 0.8)
        t_from = frame - part if frame - part > 0 else 0
        t_until = frame + 1

        # print(t_from, t_until, self.neuron_potentials.shape)
        # print(self.neuron_potentials[0, t_from:t_until])

        for i, neuron in enumerate(self.neuron_labels):
            # inputs
            if self.neuron_axes[i][0]:
                ax = self.neuron_axes[i][0]
                ax.set_xlim(-self.delta_lim + t_from, self.tacts + self.delta_lim + t_from)
                ax.xaxis.set_ticks(np.arange(t_from, t_from + self.tacts, 10))

                neuron_output = np.where(self.neuron_out[neuron, t_from:t_until] > 0.5)[0]
                self.event_plots[i].set_positions(neuron_output + t_from)

            # potentials
            if self.neuron_axes[i][1]:
                ax = self.neuron_axes[i][1]

                # ax.plot(self.neuron_potentials[neuron, t_from:t_until], '.-',color='tomato')
                ax.set_xlim(-self.delta_lim + t_from, self.tacts + self.delta_lim + t_from)
                # ax.set_ylim(-1, 5)
                ax.xaxis.set_ticks(np.arange(t_from, t_from + self.tacts, 10))
                self.lines[i].set_data(range(t_from, t_until), self.neuron_potentials[neuron, t_from:t_until])

            # outputs
            if self.neuron_axes[i][2]:
                ax = self.neuron_axes[i][2]
                ax.set_xlim(-self.delta_lim + t_from, self.tacts + self.delta_lim + t_from)
                ax.xaxis.set_ticks(np.arange(t_from, t_from + self.tacts, 10))

                neuron_output = np.where(self.neuron_outputs[neuron, t_from:t_until] > 0.5)[0]
                self.event_plots[i].set_positions(neuron_output + t_from)

    def run(self):
        self._anim_is_running = True
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.animator = FuncAnimation(self.fig, self._animate, frames=self.neuron_potentials.shape[1],
                                 init_func=self._init, interval=50, repeat=False)

        plt.show()




def animate_net_test():
    gr = nx.DiGraph()
    gr.add_nodes_from(range(0, 10))
    gr.add_edges_from([(0, 1), (1, 2), (1, 4), (1, 3), (2, 4), (3, 4), (3, 5)])
    gr.add_edges_from([(1, o) for o in range(6, 10)])
    gr.add_edges_from([(o, 4) for o in range(6, 10)])

    snn = SpikeNN(gr, [0], [5, 4], tau=1.0, threshold=1.2, dt=0.1, rest=-0.2)
    data = DataGenerator.bernoulli(1, 1, 110, 0.2)
    result, inputs_log, neurons_log, outputs_log = snn.forward(torch.from_numpy(data), logging=True)

    net_visualizator = SSNetVisualizer(snn, neurons_log[0], outputs_log[0], neurons_labels=[0, 8, 9], edge_animation=False)
    net_visualizator.run()
    # animate_net(snn, neurons_log[0], outputs_log[0])

if __name__ == '__main__':
    animate_net_test()
