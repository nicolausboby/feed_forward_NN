import random
import numpy as np
import math


class Node:
    weights = None
    n_weights = 0
    output = -1
    bias = 1

    def __init__(self, n_weights=4):
        # Initiate the weights with random value
        self.weights = []
        self.n_weights = n_weights
        for i in range(n_weights):
            self.weights.append(random.random())

    def set_output(self, output):
        self.output = output

    def __str__(self):
        strpr = ""
        for i in range(self.n_weights):
            strpr += "\n" + "weight " + str(i) + " = " + str(self.weights[i])
        return "n_weights = " + str(self.n_weights) + strpr


class Layer:
    # Number of nodes in one Layer
    nodes = []
    nb_nodes = 0

    def __init__(self, nb_nodes):
        self.nb_nodes = nb_nodes
        for i in range(nb_nodes):
            self.nodes.append(Node())

    def __str__(self):
        return "Layer :\n  > nb_nodes = " + str(self.nb_nodes)


class FeedForwardNeuralNetwork:
    layers = []
    nb_layers = 0
    nb_nodes = 0
    l_r = 0.01  # Constant
    momentum = 0.01  # Constant

    def __init__(self, nb_layers=1, nb_nodes=2):
        self.nb_layers = nb_layers
        self.nb_nodes = nb_nodes
        for i in range(nb_layers):
            self.layers.append(Layer(nb_nodes))

    def __str__(self):
        return "FFNN : \n  > nb_layers = " + str(self.nb_layers)

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

    def feed_forward(self, inputs):
        prev_outputs = inputs

        for layer in self.layers:
            new_outputs = []
            for node in layer.nodes:
                # Get sigma
                sigma = 1 * node.bias
                for i in range(len(node.weights)):
                    sigma = sigma + (prev_outputs[i] * node.weights[i])

                output = self.sigmoid(sigma)  # Use sigmoid activation
                new_outputs.append(output)  # Append result to new_outputs
                node.set_output(output) # set the output as the node's output

            prev_outputs = new_outputs.copy()


# Test
FFNN = FeedForwardNeuralNetwork()
print(FFNN.__str__())
for layer in FFNN.layers:
    print(layer.__str__())
    for node in layer.nodes:
        print(node.__str__())