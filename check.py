import numpy as np
import sfpy


from Tensor import Tensor


def load_from_file(matrix, file_object, n):
    i = 0
    for line in file_object:
        words = line.split()
        for j in range (0, n):
            matrix[i][j] = sfpy.float.Float16(words[j])
        i = i+1
    return matrix

n = 16
file_A = open('inputs/A.txt', "r")
file_B = open('inputs/B.txt', "r")
file_C = open('inputs/C.txt', "r")
a = load_from_file(np.zeros([n, n]), file_A, n)
b = load_from_file(np.zeros([n, n]), file_B, n)
c = load_from_file(np.zeros([n, n]), file_C, n)

tensor = Tensor()
d = tensor.mul(a, b, c)

"""
d1 = np.zeros([n, n])
for idx in range(2):
    b1 = np.zeros([n, n])
    c1 = np.zeros([n, n])
    b1[:, 0:8] = b[:, idx * 8: idx * 8 + 8]
    c1[:, 0:8] = c[:, idx * 8: idx * 8 + 8]
    d2 = tensor.mul(a, b1, c1)
    d1[:, idx * 8: idx * 8 + 8] = d2[:, 0:8]
print(d1)
"""

d1 = np.zeros([n, n])
b1 = np.zeros([n, n])
c1 = np.zeros([n, n])
for idx in range(4):
    b1[:, idx * 4: idx * 4 + 4] = b[:, 0:4]
    c1[:, idx * 4: idx * 4 + 4] = c[:, 0:4]
d2 = tensor.mul(a, b1, c1)
print('after')
print(d2)

"""
diff = False
print('++++++++++++++++++')
for i, row in enumerate(d1):
    for j, column in enumerate(row):
        if not column == d[i][j]:
            diff = True
            print('Diff [{}][{}]'.format(i, j))
            print('Expected value {}, Value {}'.format(column, d[i][j]))
if not diff:
    print('Matrix are equals')
"""
