import numpy as np
import sfpy


from Tensor import Tensor


def load_from_file(matrix, file_object, n):
    i = 0
    for line in file_object:
        words = line.split()
        for j in range(0, n):
            matrix[i][j] = sfpy.float.Float16(words[j])
        i = i + 1
    return matrix


n = 16
file_A = open("inputs/A.txt", "r")
file_B = open("inputs/B.txt", "r")
file_C = open("inputs/C.txt", "r")
a = load_from_file(np.zeros([n, n]), file_A, n)
b = load_from_file(np.zeros([n, n]), file_B, n)
c = load_from_file(np.zeros([n, n]), file_C, n)

tensor = Tensor(0)
d = tensor.mul(a, b, c)
d_np = np.matmul(a, b) + c

# tensor.save_files()
diff = False
print("++++++++++++++++++")
for i, row in enumerate(d_np):
    for j, column in enumerate(row):
        if not column == d[i][j]:
            diff = True
            print("Diff [{}][{}]".format(i, j))
            print("Expected value {}, Value {}".format(column, d[i][j]))
if not diff:
    print("Matrix are equals")
