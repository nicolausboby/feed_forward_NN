import neural_network
from neural_network.helper import read_csv, label_encoding, one_hot_encoding, accuracy

# Data Reading
data = read_csv("weather.csv")
# print(data)

X = []
for item in data:
    X.append(item[:-1])

y = []
for item in data:
    y.append(item[-1])

# Data Preprocessing
# print(X)
# print(y)
# print("Label Encoding")
# print("encode X")
one_hot_encoding(X)
# print("encode Y")
label_encoding(y)
# print(X)
# print(y)

model = neural_network.FeedForwardNeuralNetwork(hidden_layers=2, nb_nodes=[1,1])

# print("\nBEFORE FITTING")
# print(model)
# for layer in model.layers:
#     print(layer)
#     print(len(layer.nodes))
#     for node in layer.nodes:
#         print(node)
# print("output :", model._output_layer.nodes[0].output)

# model.fit(X, y)
model.mb_gradient_descent(inputs=X, targets=y, batch_size=2, epoch=5)

print("\n\nAFTER FITTING")
for layer in model.layers:
    print(layer)
    print(len(layer.nodes))
    for node in layer.nodes:
        print(node)
print("output :", model._output_layer.nodes[0].output)

print("\n\nPREDICTION (Using Weather Dataset)\n")
outputs = []
for i in range(len(X)):
    # print(X[i])
    outputs.append(round(model.predict(X[i])))
print(outputs)
print(y)
print("Accuracy : " + str(accuracy(outputs, y)))
