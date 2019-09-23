import random
import numpy as np
import math

class Node:
    weights = None
    n_weights = 0
    bias = 1

    def __init__(self, n_weigths = 4):
        # Initiate the weights with random value
        self.weights = []
        self.n_weigths = n_weigths
        for i in range(n_weigths):
            self.weights.append(random.random())

    def __str__(self):
        strpr = ""
        for i in range(self.n_weigths):
            strpr +=  "\n" + "weigth " + str(i) + " = " + str(self.weights[i])
        return ("n_weigths = " + str(self.n_weigths) + strpr)


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
    nb_layers = 0

    def __init__(self, nb_layers):
        self.nb_layers = nb_layers

    def predict(self, features, weights):
        predictions = np.dot(features, weights)
        return predictions

    def cost_function(self, features, targets, weights):
        """
        :param features:
        :param targets:
        :param weights:
        :return: average squared error among predictions
        """
        N = len(targets)
        predictions = self.predict(self, features, weights)
        squared_error = (predictions - targets)**2
        return 1.0/(2*N) * squared_error.sum()

    def mb_gradient_descent(self, inputs, batch_size = 32, hidden_layer = 1, nb_nodes = 2, l_r = 0.01, momentum = 0.01, epoch = 10):
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

        def sigmoid(x):
            """
            :param x: a numeric value
            :return: sigmoid value of x
            """
            return 1 / (1 + math.exp(-x))

        for i in range(epoch):
            # Do Mini Batch SGD here
            np.random.shuffle(inputs) # just dummy