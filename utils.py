

def store(
        matrix,
        filename
    ):
    with open(filename, 'w') as file:
        for row in matrix:
            for column in matrix[row]:
                file.write('{},'.format(matrix[row][column]))
            file.write('\n')
