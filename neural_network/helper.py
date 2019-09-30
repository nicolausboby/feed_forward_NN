import numpy as np
import math


def sigmoid(x):
    """
    :param x: a numeric value
    :return: sigmoid value of x
    """
    return 1 / (1 + math.exp(-x))


def dot(K, L):
    if len(K) != len(L):
        print("Length is not same, Len K : " + str(len(K)) + ", Len L : " + str(len(L)))
        print("K : " + str(K))
        print("L : " + str(L))
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))


def accuracy(outputs, targets):
    if len(outputs) != len(targets):
        print("Outputs and targets size don't match")
        return 0
    else:
        same = 0
        for i in range(len(outputs)):
            if outputs[i] == targets[i]:
                same += 1

        return same / len(targets)


def read_csv(filename, return_type="list"):
    f = open(filename, 'r')

    lines = f.read().splitlines()
    matrix = []
    for line in lines:
        row = []
        tokens = line.split(",")
        for token in tokens:
            if token.isdigit():
                row.append(float(token))
            else:
                row.append(token)
        matrix.append(row)

    return matrix


def label_encoding(X):
    # if X is multidimensional
    if isinstance(X[0], list):
        print("list")
        for col in range(len(X[0])):
            # Dictionary to store class number (0 -> n_classes-1)
            # Reset per col
            class_number = {}
            num = 0

            # Iterate per row to find classes and replace categorical to numerical inplace
            for row in range(len(X)):
                if type(X[row][col]) is not str:
                    break
                else:
                    if X[row][col] in class_number:
                        X[row][col] = class_number[X[row][col]]
                    else:
                        class_number[X[row][col]] = num
                        X[row][col] = num
                        num = num + 1
    else:
        # if X is one-dimensional
        if type(X[0]) is str:
            class_number = {}
            num = 0
            for row in range(len(X)):
                if X[row] in class_number:
                    X[row] = class_number[X[row]]
                else:
                    class_number[X[row]] = num
                    X[row] = num
                    num = num + 1


def one_hot_encoding(X):
    new_columns = dict()
    classes = dict()

    # if X is multidimensional
    if isinstance(X[0], list):
        for col in range(len(X[0])):
            col_classes = []

            # Iterate per row to find classes and replace categorical to numerical inplace
            for row in range(len(X)):
                if type(X[row][col]) is not str:
                    break
                else:
                    if X[row][col] in col_classes:
                        new_columns[X[row][col]][row] = 1
                    else:
                        col_classes.append(X[row][col])
                        new_columns[X[row][col]] = [0] * len(X)
            if col_classes:
                classes[col] = col_classes

        keys = classes.keys()
        for key in sorted(keys, reverse=True):
            for row in X:
                del row[key]

        new_X = []
        for key in new_columns:
            new_X.append(new_columns[key])

        new_X = (np.array(new_X)).T
        new_X = new_X.tolist()
        for i, row in enumerate(X):
            row.extend(new_X[i])
    else:
        # if X is one-dimensional
        if type(X[0]) is str:
            col_classes = []
            for row in range(len(X)):
                if X[row] in col_classes:
                    new_columns[X[row]][row] = 1
                else:
                    col_classes.append(X[row])
                    new_columns[X[row]] = [0] * len(X)
        if col_classes:
            classes[0] = col_classes

        X = []
        new_X = []
        for key in new_columns:
            new_X.append(new_columns[key])

        X = new_X
