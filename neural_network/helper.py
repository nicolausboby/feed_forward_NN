


def read_csv(filename, return_type = "list"):
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


