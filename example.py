import neural_network
import pandas as pd
import numpy as np

model = neural_network.FeedForwardNeuralNetwork(hidden_layers=2, nb_nodes=[1,1])

# data = pd.read_csv("weather.csv")
# print(data)

X = [[1, 80, 85.1, 1], [2, 83, 86, 2]]
y = [0, 1]

# FOR USUAL FIT
print(model)
for layer in model.layers:
    print(layer)
    print(len(layer.nodes))
    for node in layer.nodes:
        print(node)
print("output :", model._output_layer.nodes[0].output)

# model.fit(X, y)
model.mb_gradient_descent(inputs=X, targets=y, batch_size=2, epoch=5)
print("\nAFTER FITTING")
for layer in model.layers:
    print(layer)
    print(len(layer.nodes))
    for node in layer.nodes:
        print(node)
print("output :", model._output_layer.nodes[0].output)

