import os
import numpy as np
from sfpy import *

from common import debug_print, HMMA_INTS, SOURCE_INTS
from config import config


from FaultInjector import FaultInjector
from RegisterFile import RegisterFile
from TensorBuffer import TensorBuffer


class Tensor(FaultInjector):
    def __init__(self):
        super().__init__()
        config_parameters = config()['DEFAULT']
        self._threads_per_warp: int = int(config_parameters['threads_per_warp'])
        self._total_tensor_buffer: int = int(config_parameters['tensor_buffer'])
        self._thread_groups: int = int(config_parameters['thread_groups'])
        self._type: str = config_parameters['type'].lower()
        self._arch: str = config_parameters['arch'].lower()
        self._register_files = []
        self._tensor_buffer = []
        self._output_tensor_buffer = []
        self._d = []
        self._init_registers()

    def mul(self, a, b, c):
        debug_print(
            'Matrix A:\n{}\nMatrix B:\n{}\nMatrix C:\n{}'
            .format(a, b, c)
        )
        self._fill_register_files(a, b, c)
        for register, register_file in enumerate(self._register_files):
            debug_print('Register file {}\n{}'.format(
                register,
                register_file
            ))
        debug_print('Starting tensor operation')
        for instruction in range(len(HMMA_INTS)):
            debug_print('======>>> executing instruction: {}'.format(instruction))
            for thread_group in range(self._thread_groups):
                self._fill_tensor_buffer(thread_group, HMMA_INTS[instruction], instruction)
                for idx, tensor_buffer in enumerate(self._tensor_buffer):
                    debug_print('Tensor Buffer Data Before {}\n{}'.format(
                            idx,
                            tensor_buffer
                        )
                    )
                self._execution(thread_group, instruction)
        for register, register_file in enumerate(self._register_files):
            debug_print('Register File After {}\n{}'.format(
                    register,
                    register_file
                )
            )
        self._d = self._read_result()
        return self._d
    
    def save_files(self):
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        for idx, tensor_buffer in enumerate(self._tensor_buffer):
            filename = 'outputs/tensor_buffer_{}.txt'.format(idx)
            tensor_buffer.store(filename)
        for idx, register_file in enumerate(self._register_files):
            filename = 'outputs/register_file_{}.txt'.format(idx)
            register_file.store(filename)
        datafile = open('outputs/result.txt', 'w')
        for row in self._d:
            for column in row:
                datafile.write('{}\n'.format(column))
        datafile.close()


    def _read_result(self):
        data = []
        for register, register_file in enumerate(self._register_files):
            data.append([])
            for line in range(4):
                data[register].extend(register_file.rf_read(line + 4))
        matrix = np.zeros([16, 16])
        for thread_id in range(self._threads_per_warp):
            if thread_id < 4:
                for i in range(8):
                    matrix[thread_id][i] = data[thread_id][i]
            elif thread_id < 8:
                for i in range(8):
                    matrix[thread_id + 4][i] = data[thread_id][i]
            elif thread_id < 12:
                for i in range(8):
                    matrix[thread_id - 8][i + 8] = data[thread_id][i]
            elif thread_id < 16:
                for i in range(8):
                    matrix[thread_id - 4][i + 8] = data[thread_id][i]
            elif thread_id < 20:
                for i in range(8):
                    matrix[thread_id - 12][i] = data[thread_id][i]
            elif thread_id < 24:
                for i in range(8):
                    matrix[thread_id - 8][i] = data[thread_id][i]
            elif thread_id < 28:
                for i in range(8):
                    matrix[thread_id - 20][i + 8] = data[thread_id][i]
            else:
                for i in range(8):
                    matrix[thread_id - 16][i + 8] = data[thread_id][i]
        
        return matrix

    def _execution(self, thread_group, inst):
        switcher = {
            'volta': self._volta,
            'pascal': self._pascal,
            'turing': self._turing
        }
        func = switcher.get(self._arch, lambda: [])
        return func(thread_group, inst)
    
    def _fill_register_files(self, a, b, c):
        pointer = [
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        debug_print('Filling the register file on the GPU core')
        if self._threads_per_warp > 32:
            raise Exception('Error addressing the RF and locate operands')
        for thread_id in range(self._threads_per_warp):
            debug_print('Storing data for {} thread\n'.format(thread_id))
            if thread_id < 4:
                pointer[0][0] = 0
                pointer[0][1] = thread_id

                pointer[1][0] = thread_id
                pointer[1][1] = 0

                pointer[2][0] = 0
                pointer[2][1] = thread_id
            elif thread_id < 8:
                pointer[0][0] = 0
                pointer[0][1] = thread_id + 4

                pointer[1][0] = thread_id - 4
                pointer[1][1] = 0

                pointer[2][0] = 0
                pointer[2][1] = thread_id + 4
            elif thread_id < 12:
                pointer[0][0] = 0
                pointer[0][1] = thread_id - 8

                pointer[1][0] = thread_id
                pointer[1][1] = 0

                pointer[2][0] = 8
                pointer[2][1] = thread_id - 8
            elif thread_id < 16:
                pointer[0][0] = 0
                pointer[0][1] = thread_id - 4

                pointer[1][0] = thread_id - 4
                pointer[1][1] = 0

                pointer[2][0] = 8
                pointer[2][1] = thread_id - 4
            elif thread_id < 20:
                pointer[0][0] = 0
                pointer[0][1] = thread_id - 12

                pointer[1][0] = thread_id -12
                pointer[1][1] = 0

                pointer[2][0] = 0
                pointer[2][1] = thread_id - 12
            elif thread_id < 24:
                pointer[0][0] = 0
                pointer[0][1] = thread_id - 8

                pointer[1][0] = thread_id - 16
                pointer[1][1] = 0

                pointer[2][0] = 0
                pointer[2][1] = thread_id - 8
            elif thread_id < 28:
                pointer[0][0] = 0
                pointer[0][1] = thread_id - 20

                pointer[1][0] = thread_id - 12
                pointer[1][1] = 0

                pointer[2][0] = 8
                pointer[2][1] = thread_id - 20
            else:
                pointer[0][0] = 0
                pointer[0][1] = thread_id - 16

                pointer[1][0] = thread_id - 16
                pointer[1][1] = 0

                pointer[2][0] = 8
                pointer[2][1] = thread_id - 16
        
            enable_c = True

            source = SOURCE_INTS[0][1:]
            debug_print('Pointers1:\n {}\n'.format(pointer))
            debug_print('Source1:\n {}\n'.format(source))
            self._register_files[thread_id].fill(
                a, b, c, pointer, source, enable_c
            )

            pointer[0][0] += 4
            pointer[1][1] += 4
            pointer[2][0] += 4

            source = SOURCE_INTS[1][1:]
            debug_print('Pointers2:\n {}\n'.format(pointer))
            debug_print('Source2:\n {}\n'.format(source))
            self._register_files[thread_id].fill(
                a, b, c, pointer, source, enable_c
            )

            pointer[0][0] += 4
            pointer[1][1] += 4

            enable_c = False
            source = SOURCE_INTS[2][1:]
            debug_print('Pointers3:\n {}\n'.format(pointer))
            debug_print('Source3:\n {}\n'.format(source))
            self._register_files[thread_id].fill(
                a, b, c, pointer, source, enable_c
            )

            pointer[0][0] += 4
            pointer[1][1] += 4

            source = SOURCE_INTS[3][1:]
            debug_print('Pointers4:\n {}\n'.format(pointer))
            debug_print('Source4:\n {}\n'.format(source))
            self._register_files[thread_id].fill(
                a, b, c, pointer, source, enable_c
            )


    def _init_registers(self):
        for _ in range(self._threads_per_warp):
            self._register_files.append(RegisterFile())
        for _ in range(self._total_tensor_buffer):
            self._tensor_buffer.append(TensorBuffer())
            self._output_tensor_buffer.append(TensorBuffer())

    
    def _fill_tensor_buffer(
            self,
            thread_group,
            instruction_sources,
            instruction_number
        ):
        debug_print(
            'Starting filling tensor buffer {} thread group'
            .format(thread_group)
        )
        pointer_storage = 0 if thread_group < 4 else 1
        pointer_tensor_buffer = thread_group % 4
        debug_print('Instruction sources {}'.format(instruction_sources))
        debug_print('Tensor buffer pointer {}'.format(pointer_tensor_buffer))

        for i in range(4):
            if instruction_number % 2 == 0:
                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[1])
                address = 'a_{}0'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'A',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'a_{}1'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'A',
                    address,
                    values[1],
                    pointer_storage
                )

                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[1] + 1)
                address = 'a_{}2'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'A',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'a_{}3'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'A',
                    address,
                    values[1],
                    pointer_storage
                )
                
                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[2])
                address = 'b_0{}'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'B',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'b_1{}'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'B',
                    address,
                    values[1],
                    pointer_storage
                )

                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[2] + 1)
                address = 'b_2{}'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'B',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'b_3{}'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'B',
                    address,
                    values[1],
                    pointer_storage
                )

                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[3])
                address = 'c_{}0'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'c_{}1'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[1],
                    pointer_storage
                )

                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[3] + 1)
                address = 'c_{}2'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'c_{}3'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[1],
                    pointer_storage
                )
            if instruction_number % 2 == 1:
                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[3])
                address = 'c_{}0'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'c_{}1'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[1],
                    pointer_storage
                )

                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[3] + 1)
                address = 'c_{}2'.format(i)
                debug_print('Values: {}'.format(values))
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[0],
                    pointer_storage
                )
                address = 'c_{}3'.format(i)
                debug_print('Address {}'.format(address))
                debug_print('Pointer tensor {}'.format(pointer_tensor_buffer))
                self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                    'C',
                    address,
                    values[1],
                    pointer_storage
                )

    def _fma_core(self, a, b, c):
        return a * b + c
    
    def _tensor_ele(self,a, b, c, n):
        w0, w1, w2, w3 = 0.0, 0.0, 0.0, 0.0
        if len(a) > n and len(b) > n:
            raise Exception(
                'Error using the tensor element...'
            )
        w0 = self._fma_core(a[0], b[0], c)
        w1 = self._fma_core(a[1], b[1], w0)
        w2 = self._fma_core(a[2], b[2], w1)
        w3 = self._fma_core(a[3], b[3], w2)
        return w3
    
    def _volta(self, thread_group, inst):
        pointer_tensor_buffer = thread_group % 4

        debug_print('Thread group {}'.format(thread_group))
        debug_print('Pointer Tensor Buffer {}'.format(pointer_tensor_buffer))
        debug_print('Instruction {}'.format(inst))

        A = np.zeros([16, 4])
        B = np.zeros([16, 4])
        C = np.zeros(16)
        W = np.zeros([4, 4])

        if inst % 2 == 0 and thread_group < 4:
            buffer_c_des = 0
            buffer_c_group = 0
            pointer_a = 0
            pointer_b = 0
            pointer_c = 0
        elif inst % 2 == 1 and thread_group < 4:
            buffer_c_des = 0
            buffer_c_group = 1
            pointer_a = 0
            pointer_b = 1
            pointer_c = 0
        elif inst % 2 == 0 and thread_group >= 4:
            buffer_c_des = 1
            buffer_c_group = 0
            pointer_a = 1
            pointer_b = 0
            pointer_c = 1
        elif inst % 2 == 1 and thread_group >= 4:
            buffer_c_des = 1
            buffer_c_group = 1
            pointer_a = 1
            pointer_b = 1
            pointer_c = 1

        for k in range(4):
            for i in range(4):
                for j in range(4):
                    A[k * 4 + i][j] = self._tensor_buffer[pointer_tensor_buffer].read_buffer(
                        'A',
                        'a_{}{}'.format(k, j),
                        pointer_a
                    )
                    B[k * 4 + i][j] = self._tensor_buffer[pointer_tensor_buffer].read_buffer(
                        'B',
                        'b_{}{}'.format(j, i),
                        pointer_b
                    )
                C[k * 4 + i] = self._tensor_buffer[pointer_tensor_buffer].read_buffer(
                    'C',
                    'c_{}{}'.format(k, i),
                    pointer_c
                )
        
        debug_print('A execution:\n{}'.format(A))
        debug_print('B execution:\n{}'.format(B))
        debug_print('C execution:\n{}'.format(C))

        for k in range(4):
            for i in range(4):
                W[k][i] = self._dot_product(
                    A[k * 4 + i],
                    B[k * 4 + i],
                    C[k * 4 + i]
                )
        debug_print('Values on the tensor units:\n{}'.format(W))

        for i in range(4):
                for j in range(4):
                    if buffer_c_group == 0:
                        self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                            'C',
                            'c_{}{}'.format(i, j),
                            W[i][j],
                            buffer_c_des
                        )
                    elif buffer_c_group == 1:
                        self._tensor_buffer[pointer_tensor_buffer].buffer_write(
                            'CX',
                            'c_{}{}'.format(i, j),
                            W[i][j],
                            buffer_c_des
                        )

        for i in range(4):
            for j in range(2):
                data = [W[i][j * 2], W[i][j * 2 + 1]]
                self._register_files[thread_group * 4 + i].rf_write(
                    HMMA_INTS[inst][0] + j,
                    data
                )
    
    def _pascal(self, a, b, c, thread_group):
        w = [ [0.0] * 16 ] * 16
        d = [ [0.0] * 16 ] * 16

        debug_print('A: {}\nB: {}\nC: {}'.format(a[0][0], b[0][0], c[0][0]))
        w[0][0] = self._fma_core(a[0][0], b[0][0], c[0][0])
        w[0][1] = self._fma_core(a[0][0], b[0][1], c[0][1])
        w[0][2] = self._fma_core(a[0][0], b[0][2], c[0][2])
        w[0][3] = self._fma_core(a[0][0], b[0][3], c[0][3])

        for k in range(4):
            for i in range(1, 4):
                for j in range(4):
                    w[i][j] = self._fma_core(a[0][i], b[i][j], w[i-1][j])
                    w[i][j] = self._fma_core(a[0][i], b[i][j], w[i-1][j])
                    w[i][j] = self._fma_core(a[0][i], b[i][j], w[i-1][j])
                    w[i][j] = self._fma_core(a[0][i], b[i][j], w[i-1][j])

            for i in range(4):
                d[k][i] = w[3][i]
                d[k][i] = w[3][i]
                d[k][i] = w[3][i]
                d[k][i] = w[3][i]

        return d
    
    def _turing(self, a,  b,  c, thread_group, n):
        d = []
        for i in range(4):
            for j in range(4):
                d[i][j] = self._tensor_ele(a[i], b[j], c[i][j], n)
        
        return d
    
    def _dot_product(self, a,  b,  c):
        if not len(a) == len(b):
            raise Exception(
                'Error using the tensor element...'
            )
        if self._type == 'float16':
            dot_products = np.zeros([4], dtype=Float16)
            for i in range(4):
                dot_products[i] = Float16(a[i]) * Float16(b[i])
            return dot_products[0] + dot_products[1] + dot_products[2] + dot_products[3] + Float16(c)
        else:
            dot_products = np.zeros([4], dtype=Posit16)
            for i in range(4):
                dot_products[i] = Posit16(a[i]) * Posit16(b[i])
            return dot_products[0] + dot_products[1] + dot_products[2] + dot_products[3] + Posit16(c)

def load_from_file(matrix, file_object, n):
    i = 0
    for line in file_object:
        words = line.split()
        for j in range (0, n):
            matrix[i][j] = words[j]
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
tensor.save_files()
print('++++++++++++++++++')
for row in d:
    for column in row:
        print(column)
