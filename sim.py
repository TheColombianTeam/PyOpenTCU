import argparse, os
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


BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'faults')
DIFF_FILE = os.path.join(RESULTS_DIR, 'diff.txt')
FAULTS_DIR = os.path.join(RESULTS_DIR, 'fault.csv')


def save_matrix(a, b, c):
    np.save('faults/a', a)
    np.save('faults/b', b)
    np.save('faults/c', c)


def load_matrix():
    a = np.load('faults/a.npy')
    b = np.load('faults/b.npy')
    c = np.load('faults/c.npy')
    return a, b, c


def deco_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('fault', default=None, type=str)
    parser.add_argument('process', default=None, type=str)
    return parser.parse_args()


def golden_simulation(a, b, c):
    save_matrix(a, b, c)
    config_parameters = config()['DEFAULT']
    type: str = config_parameters['type'].lower()
    tensor = Tensor()
    d = tensor.mul(a, b, c)
    for i, row in enumerate(d):
        for j, column in enumerate(row):
            if type == 'float16':
                input_format = Float16(column)
            else:
                input_format = Posit16(column)
            print('{}:{}:{}'.format(
                    i * 16 + j,
                    column,
                    hex(input_format.bits)
                )
            )


def run_simulation(idx=None):
    a, b, c = load_matrix()
    config_parameters = config()['DEFAULT']
    type: str = config_parameters['type'].lower()
    tensor = Tensor(idx)
    d = tensor.mul(a, b, c)
    for i, row in enumerate(d):
        for j, column in enumerate(row):
            if type == 'float16':
                input_format = Float16(column)
            else:
                input_format = Posit16(column)
            value = input_format.bits
            value_hex = hex(value)
            print(
                '{}:{}:{}'.format(
                    i * 16 + j,
                    column,
                    value_hex
                )
            )


def read_results(idx):
    fault = identify_fault(idx)
    config_parameters = config()['DEFAULT']
    type: str = config_parameters['type'].lower()
    with open(DIFF_FILE, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.split('|')
            golden = data[0].strip().split(':')
            faulty = data[1].strip().split(':')
            try:
            # if not is internal data
                aux = int(golden[0])
                if type == 'float16':
                    golden_format = Float16(golden[1])
                    faulty_format = Float16(faulty[1])
                    golden_data = golden_format
                    faulty_data = faulty_format
                    mask_golden = golden_format.from_bits(int(golden[2], 16))
                    mask_faulty = faulty_format.from_bits(int(faulty[2], 16))
                else:
                    golden_format = Posit16(golden[1])
                    faulty_format = Posit16(faulty[1])
                    golden_data = golden_format
                    faulty_data = faulty_format
                    mask_golden = golden_format.from_bits(int(golden[2], 16))
                    mask_faulty = faulty_format.from_bits(int(faulty[2], 16))
                mask_golden_bits = mask_golden.bits
                mask_faulty_bits = mask_faulty.bits
                mask = mask_golden_bits ^ mask_faulty_bits
                error_relative = relative_error(golden_format, faulty_format)
                error_abs = abs_error(golden_format, faulty_format)
                print('{},{},{},{},{},{},{},{},{}'.format(
                        fault.split('\n')[0],
                        golden[0],
                        golden_data,
                        hex(mask_golden_bits),
                        faulty_data,
                        hex(mask_faulty_bits),
                        hex(mask), 
                        error_relative, 
                        error_abs
                    )
                )
            except:
                # Analysis internal data 
                # C:15:-:1.8251953125:0x3f4d
                # W:3:3:1.505859375:0x3e06
                if type == 'float16':
                    golden_format = Float16(golden[3])
                    faulty_format = Float16(faulty[3])
                    golden_data = golden_format
                    faulty_data = faulty_format
                    mask_golden = golden_format.from_bits(int(golden[4], 16))
                    mask_faulty = faulty_format.from_bits(int(faulty[4], 16))
                else:
                    golden_format = Posit16(golden[3])
                    faulty_format = Posit16(faulty[3])
                    golden_data = golden_format
                    faulty_data = faulty_format
                    mask_golden = golden_format.from_bits(int(golden[4], 16))
                    mask_faulty = faulty_format.from_bits(int(faulty[4], 16))
                mask_golden_bits = mask_golden.bits
                mask_faulty_bits = mask_faulty.bits
                mask = mask_golden_bits ^ mask_faulty_bits
                error_relative = relative_error(golden_format, faulty_format)
                error_abs = abs_error(golden_format, faulty_format)
                print('{},{},{},{},{},{},{},{},{}'.format(
                        fault.split('\n')[0],
                        '{}_{}_{}'.format(golden[0], golden[1], golden[2]),
                        golden_data,
                        hex(mask_golden_bits),
                        faulty_data,
                        hex(mask_faulty_bits),
                        hex(mask), 
                        error_relative, 
                        error_abs
                    )
                )


def identify_fault(idx):
    faults_file = open(FAULTS_DIR, 'r')
    faults = faults_file.readlines()
    faults_file.close()
    fault = faults[idx]
    return fault


def relative_error(real, value):
    return ((value - real) / real)


def abs_error(real, value):
    return (np.abs(value - real))
      

if __name__ == '__main__':
    args = deco_arg()
    idx = None if args.fault == 'None' else int(args.fault)
    process = args.process
    if process == 'run':
        if idx == None:
            a = np.random.rand(16, 16)
            b = np.tril(np.random.rand(16, 16), -1)
            c = np.random.rand(16, 16)
            golden_simulation(a, b, c)
        else:
            run_simulation(idx)
    else:
        read_results(idx)