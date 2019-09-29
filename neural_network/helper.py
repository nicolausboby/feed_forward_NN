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


def encoding_label(X):

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
