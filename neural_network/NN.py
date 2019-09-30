import random
import numpy as np
from neural_network.helper import sigmoid
from neural_network.helper import dot


class Node(object):
    output = np.NaN
    delta = np.NaN

    def __init__(self, n_weights=4):
        # weights[0] for bias
        # weights[1..N] for inputs or previous layer
        self.weights = []
        self.n_weights = n_weights
        self.bias = 1

        for i in range(n_weights + 1):
            # Initiate the weights with random value 
            self.weights.append(random.random())

    def _update(self, n_input):
        self.n_weights = n_input
        self.weights.clear()
        for i in range(self.n_weights + 1):
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

    def set_weight(self, index, weight):
        self.weights[index] = weight


class Layer(object):
    # Number of nodes in one Layer

    def __init__(self, nb_nodes, input_len):
        # nb_nodes : number of node in the layer
        # input_len : number of input edges
        self.nodes = []
        self.nb_nodes = nb_nodes
        for i in range(nb_nodes):
            self.nodes.append(Node(input_len))

    def _update(self, input_len):
        if input_len <= 0: raise ValueError("Invalid number of input")
        for node in self.nodes:
            node._update(input_len)

    def __str__(self):
        return "Layer :\n  > nb_nodes = " + str(self.nb_nodes)


class FeedForwardNeuralNetwork(object):
    layers = []
    learning_rate = 0.01
    momentum = 0.9

    def __str__(self):
        return "FFNN : \n  > nb_layers = " + str(self._nb_layers)

    def __init__(self, hidden_layers=1, nb_nodes=None):
        if nb_nodes is None:
            nb_nodes = [1]
        if hidden_layers <= 0: raise ValueError('Invalid nb_layers')
        if len(nb_nodes) != hidden_layers:
            raise ValueError('hidden_layer and amount of item in nb_nodes are different')
        for i in nb_nodes:
            if i <= 0: raise ValueError('Invalid nb_nodes')

        self._output_layer = Layer(1, nb_nodes[-1])
        self._nb_layers = hidden_layers
        self._nb_nodes = nb_nodes
        self.layers = []

        for i, n_node in enumerate(nb_nodes):
            if i == 0:  # First hidden layer
                new_layer = Layer(nb_nodes=n_node, input_len=2)
                self.layers.append(new_layer)
            else:
                new_layer = Layer(n_node, nb_nodes[i - 1])
                self.layers.append(new_layer)


    def feed_forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i == 0:  # hidden layer[0]
                for node in layer.nodes:
                    v = dot(node.weights[1:], inputs)
                    v += node.bias * node.weights[0]
                    node.set_output(sigmoid(v))
            else:  # hidden layer[1..N]
                for node in layer.nodes:
                    y_prev = []
                    for prev_node in self.layers[i - 1].nodes:
                        y_prev.append(prev_node.output)
                    v = dot(node.weights[1:], y_prev)
                    v += node.bias * node.weights[0]
                    node.set_output(sigmoid(v))

        y_hidden = []
        for last_nodes in self.layers[-1].nodes:
            y_hidden.append(last_nodes.output)
        v_output = dot(self._output_layer.nodes[0].weights[1:], y_hidden)
        v_output += self._output_layer.nodes[0].bias * self._output_layer.nodes[0].weights[0]
        self._output_layer.nodes[0].set_output(sigmoid(v_output))

    def backward_prop(self, target):
        # For output layer
        outnode = self._output_layer.nodes[0]
        delta_outnode = outnode.output * (1 - outnode.output) * (target - outnode.output)
        self._output_layer.nodes[0].set_delta(delta_outnode)
        # For hidden layer
        for i, layer in enumerate(reversed(self.layers)):
            # Last hidden layer
            if i == 0:
                for j, node in enumerate(layer.nodes):
                    hidden_delta = node.output * (1 - node.output) * (outnode.weights[j + 1] * delta_outnode)
                    node.set_delta(hidden_delta)
            # 1..N-1 hidden layers
            else:
                if (self._nb_layers > 1):
                    for j, node in enumerate(layer.nodes):
                        # Multiply weights*delta for all nodes in next layer
                        result = 0
                        for nextlayer_node in self.layers[self._nb_layers - i].nodes:
                            result = result + (nextlayer_node.weights[j + 1] * nextlayer_node.delta)
                        hidden_delta = node.output * (1 - node.output) * result
                        node.set_delta(hidden_delta)
        return

    def update_weigths(self):
        # For hidden layers
        for layer in self.layers:
            for node in layer.nodes:
                for i, weight in enumerate(node.weights):
                    new_weight = weight + (self.momentum * weight) + (self.learning_rate * node.delta * node.output)
                    node.set_weight(i, new_weight)
        # For output layer
        for node in self._output_layer.nodes:
            for i, weight in enumerate(node.weights):
                new_weight = weight + (self.momentum * weight) + (self.learning_rate * node.delta * node.output)
                node.set_weight(i, new_weight)
        return

    def fit(self, X, y):
        """
        :param X: matrix 2D inputs
        :param y: list of inputs
        :return: sigmoid value of x
        """
        self.layers[0]._update(len(X[0]))  # reshape weights hidden layer to input

        for row in X:
            self.feed_forward(row)
            # self.backward_prop(y)
            # self.update_weigths()
        return

    def partial_fit(self, X, y, batch_size):
        """
        :param X: matrix 2D inputs (batch sized)
        :param y: targets
        :param batch_size: row size per batch
        :return:
        """

        # self.layers[0]._update(len(X[0]))  # reshape weights hidden layer to input

        # Forward pass and Backprop per row
        for i, row in enumerate(X):
            self.feed_forward(row)

            # #Test FEED FORWARD
            # for layer in self.layers:
            #     for node in layer.nodes:
            #         print("HIDLAYER OUTPUT: " + str(node.output))
            # print("OUTLAYER OUTPUT: " + str(self._output_layer.nodes[0].output))

            self.backward_prop(y[i])

            # #Test BACKPROP
            # print("OUTLAYER DELTA: " + str(self._output_layer.nodes[0].delta))
            # for layer in reversed(self.layers):
            #     for node in layer.nodes:
            #         print("HIDLAYER DELTA: " + str(node.delta))

            # Update weights per batch
            if (i + 1) % batch_size == 0:
                self.update_weigths()

            # #Test UPDATE WEIGHT
            # for layer in self.layers:
            #     for node in layer.nodes:
            #         print("HIDLAYER WEIGHTS: " + str(node.weights))
            # print("OUTLAYER WEIGHTS: " + str(self._output_layer.nodes[0].weights))

        return

    def mb_gradient_descent(self, inputs, targets, batch_size=32, epoch=10):
        """
        :param inputs: train data
        :param batch_size: Size of batch per weight update
        :param epoch: number of iteration(s)
        :return: trained model
        """
        self.layers[0]._update(len(inputs[0]))  # reshape weights hidden layer to input

        for i in range(epoch):
            self.partial_fit(inputs, targets, batch_size)

    def predict(self, inputs):
        # DARI FEED FORWARD
        # for i, layer in enumerate(self.layers):
        #     if i == 0:  # hidden layer[0]
        #         for node in layer.nodes:
        #             v = dot(node.weights[1:], inputs)
        #             v += node.bias * node.weights[0]
        #             node.set_output(sigmoid(v))
        #     else:  # hidden layer[1..N]
        #         for node in layer.nodes:
        #             y_prev = []
        #             for prev_node in self.layers[i - 1].nodes:
        #                 y_prev.append(prev_node.output)
        #             v = dot(node.weights[1:], y_prev)
        #             v += node.bias * node.weights[0]
        #             node.set_output(sigmoid(v))
        #
        # y_hidden = []
        # for last_nodes in self.layers[-1].nodes:
        #     y_hidden.append(last_nodes.output)
        # v_output = dot(self._output_layer.nodes[0].weights[1:], y_hidden)
        # v_output += self._output_layer.nodes[0].bias * self._output_layer.nodes[0].weights[0]
        # return sigmoid(v_output)

        y_prev = inputs
        y_temp = []

        for i, layer in enumerate(self.layers):
            for node in layer.nodes:
                v = dot(node.weights[1:], y_prev)
                v += node.bias * node.weights[0]
                y_temp.append(sigmoid(v))

            y_prev = y_temp.copy()
            y_temp.clear()

        y_hidden = y_prev.copy()
        v_output = dot(self._output_layer.nodes[0].weights[1:], y_hidden)
        v_output += self._output_layer.nodes[0].bias * self._output_layer.nodes[0].weights[0]

        return sigmoid(v_output)
