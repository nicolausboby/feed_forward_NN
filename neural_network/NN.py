import random
import numpy as np
import math
import pandas as pd


class Node(object):
    def __init__(self, n_weights=4):
        # weights[0] for bias
        # weights[1..N] for inputs or previous layer
        self.weights = []
        self.n_weights = n_weights
        self.bias = 1

        for i in range(n_weights + 1):
            # Initiate the weights with random value    
            self.weights.append(random.random())

    def __call__(self, n_input):
        self.n_weights  = n_input
        for i in range(n_weights + 1):
            self.weights.append(random.random())

    def __str__(self):
        strpr = ""
        for i in range(self.n_weights):
            strpr += "\n" + "weight " + str(i) + " = " + str(self.weights[i])
        return "n_weights = " + str(self.n_weights) + strpr

    def __repr__(self):
        return self.__str__()

    def set_output(self, output):
        self.output = output

    def set_delta(self, delta):
        self.delta = delta



class Layer:
    # Number of nodes in one Layer
    nodes = []
    nb_nodes = 0

    def __init__(self, nb_nodes, input_len=1):
        self.nb_nodes = nb_nodes
        for i in range(nb_nodes):
            self.nodes.append(Node(input_len))

    def __call__(self, input_len):
        if input_len <= 0: raise ValueError("Invalid number of input")
        for node in nodes:
            node(input_len)

    def __str__(self):
        return "Layer :\n  > nb_nodes = " + str(self.nb_nodes)


class FeedForwardNeuralNetwork:
    layers = []
    learning_rate = 0.01
    momentum = 0.01

    def feed_forward(self, X):
        for layer in layers:
            for node in layer:
                pass
        return

    def __init__(self, hidden_layers=1, nb_nodes=[1]):
        if hidden_layers <= 0: raise ValueError('Invalid nb_layers')
        if len(nb_nodes) != hidden_layers: 
            raise ValueError('hidden_layer and amount of item in nb_nodes are different')
        for i in nb_nodes:
            if i <= 0: raise ValueError('Invalid nb_nodes')

        self._output_layer = Layer(1)
        self._nb_layers = hidden_layers
        self._nb_nodes = nb_nodes
        self.layers = []

        for i, n_node in enumerate(nb_nodes):
            self.layers.append(Layer(n_node))

    def fit(self, X, y):
        return

    def predict(self):
        return

    def __str__(self):
        return "FFNN : \n  > nb_layers = " + str(self._nb_layers)


    def sigmoid(self, x):
        """
        :param x: a numeric value
        :return: sigmoid value of x
        """
        return 1 / (1 + math.exp(-x))


    def cost_function(self, features, targets, weights):
        """
        :note : Should be done only by output node (1 only)
        :param features: Attributes from data
        :param targets: actual result from features
        :param weights: Coefficients (retrieved from notes)
        :return: average squared error among predictions
        """
        N = len(targets)
        predictions = np.dot(features, weights)
        squared_error = (targets - predictions) ** 2
        return 1.0 / (2 * N) * squared_error.sum()


    def mb_gradient_descent(self, inputs, batch_size= 32, epoch=10):
        """
        :param inputs: train data
        :param batch_size: Size of batch per weight update
        :param hidden_layer: Number of hidden layer(s) inside NN
        :param nb_nodes: Number of nodes per hidden layer
        :param l_r: Size of steps taken
        :param momentum:
        :param epoch: number of iteration(s)
        :return: trained model
        """

        for i in range(epoch):
            # Do Mini Batch SGD here
            np.random.shuffle(inputs)  # just dummy



        

    def backward_prop(self, target):
        for i, layer in enumerate(reversed(self.layers)):
            if i == self.nb_layers - 1:
                for node in layer.nodes:
                    delta = node.output * (1 - node.output) * (target - node.output)
                    node.set_delta(delta)
            else:
                for j, node in enumerate(layer.nodes):
                    out_node = self.layers[-1].nodes[0]
                    delta = node.output * (1 - node.output) * (out_node.weights[j+1] * out_node.delta)
                    node.set.delta(delta)
        return


# Test
FFNN = FeedForwardNeuralNetwork()
print(FFNN.__str__())
for layer in FFNN.layers:
    print(layer.__str__())
    for node in layer.nodes:
        print(node.__str__())