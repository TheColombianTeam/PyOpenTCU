import numpy as np
from sfpy import *


from config import config
from Tensor import Tensor


"""
    target: input(0), output(1), interconnection(2)
    target_thread_group max 8
    position
    mask
    type: bf, sa0, sa1
"""


def save_results(d, filename):
    config_parameters = config()['DEFAULT']
    type: str = config_parameters['type'].lower()
    with open(filename, 'w') as file:
        for i, row in enumerate(d):
            for j, column in enumerate(row):
                if type == 'float16':
                    input_format = Float16(column)
                else:
                    input_format = Posit16(column)
                file.write('[{}, {}]: {} <-> {}\n'.format(
                        i * 16, j,
                        column,
                        hex(input_format.bits)
                    )
                )


def golden_simulation(a, b, c):
    tensor = Tensor()
    d = tensor.mul(a, b, c)
    save_results(d, 'faults/golden.txt')


def run_simulation(a, b, c):
    faults = open('faults/fault.csv', 'r')
    total_faults = faults.readlines()
    faults.close()
    for idx in range(len(total_faults)):
        tensor = Tensor(idx)
        d = tensor.mul(a, b, c)
        save_results(d, 'faults/fault_{}.txt'.format(idx))


if __name__ == '__main__':
    a = np.random.rand(16, 16)
    b = np.random.rand(16, 16)
    c = np.random.rand(16, 16)
    golden_simulation(a, b, c)
    run_simulation(a, b, c)
