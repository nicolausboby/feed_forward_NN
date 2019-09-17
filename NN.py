import random
import numpy as np

class Node:
    weights = None
    n_weigths = 0

    def __init__(self, n_weigths = 4):
        # Initiate the weigths with random value
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
