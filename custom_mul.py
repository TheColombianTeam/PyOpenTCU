import numpy as np


from config import config
from Tensor import Tensor


MS = 128
NS = 128
KS = 8
MR = 8
NR = 8


def complete(matrix):
    shape = matrix.shape
    if (shape[0] % MS) > 0:
        new_shape_a =  MS * (shape[0] // MS) + MS
    else:
        new_shape_a =  shape[0]
    if (shape[1] % NS) > 0:
        new_shape_b =  NS * (shape[1] // NS) + NS
    else:
        new_shape_b =  shape[1]
    new_shape = new_shape_a, new_shape_b
    new_matrix = np.zeros(new_shape)
    new_matrix[:shape[0], :shape[1]] = matrix
    return new_matrix


def scheduler(a, b, c, sm):
    c_shape_original = c.shape
    a = complete(a)
    b = complete(b)
    c = complete(c)
    a_shape = a.shape
    b_shape = b.shape
    c_shape = c.shape
    print(a.shape)
    print(b.shape)
    print(c.shape)
    for row_c in range(c_shape[0] // MS):
        new_row_c_start = MS * row_c
        new_row_c_end = MS * row_c + MS
        for column_c in range(c_shape[1] // NS):
            new_column_c_start = NS * column_c
            new_column_c_end = NS * column_c + NS
            new_row_a_start = new_row_c_start
            new_row_a_end = new_row_c_end
            new_column_b_start = new_column_c_start
            new_column_b_end = new_column_c_end
            c_block = np.zeros([MS, NS])
            for column_a in range(a_shape[1] // KS):
                new_column_a_start = KS * column_a
                new_column_a_end = KS * column_a + KS
                new_row_b_start = new_column_a_start
                new_row_b_end = new_column_a_end
                a_block = a[new_row_a_start:new_row_a_end, new_column_a_start:new_column_a_end]
                b_block = b[new_row_b_start:new_row_b_end, new_column_b_start:new_column_b_end]
                c_block = c[new_row_c_start:new_row_c_end, new_column_c_start:new_column_c_end]
                c_block = np.matmul(a_block, b_block) + c_block
                c[new_row_c_start:new_row_c_end, new_column_c_start:new_column_c_end] = c_block
    return c[:c_shape_original[0], :c_shape_original[1]]


if __name__ == '__main__':
    a = np.random.rand(300, 100)
    b = np.random.rand(100, 300)
    c = np.random.rand(300, 300)
    d = np.matmul(a, b) + c
    c_ = scheduler(a, b, c, None)
    for i, row in enumerate(c_):
        for j, column in enumerate(row):
            if np.abs(d[i, j] - c_[i, j]) > 0.1:
                print('[{},{}]->{}:{}'.format(i, j, d[i, j], c_[i, j]))

