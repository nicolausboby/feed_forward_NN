import neural_network
from neural_network.helper import read_csv, label_encoding, one_hot_encoding, accuracy

# Data Reading
data = read_csv("weather.csv")

X = []
for item in data:
    X.append(item[:-1])

y = []
for item in data:
    y.append(item[-1])

# Data Preprocessing
one_hot_encoding(X)
label_encoding(y)

model = neural_network.FeedForwardNeuralNetwork(hidden_layers=2, nb_nodes=[2, 2])
model.mb_gradient_descent(inputs=X, targets=y, batch_size=2, epoch=5)

print("\n\nPREDICTION (Using Weather Dataset)\n")
outputs = []
for i in range(len(X)):
    outputs.append(round(model.predict(X[i])))
print(outputs)
print(y)
print("Accuracy : " + str(accuracy(outputs, y)))