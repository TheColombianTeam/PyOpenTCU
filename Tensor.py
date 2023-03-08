import numpy as np


from common import HMMA_INTS, debug_print
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
        self._arch: str = config_parameters['arch'].lower()
        self._register_files = []
        self._tensor_buffer = []
        self._output_tensor_buffer = []
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
                debug_print(self._tensor_buffer[0])
                d = self._execution(a, b, c)
        return d

    def _execution(self, a, b, c):
        switcher = {
            'volta': self._volta,
            'pascal': self._pascal,
            'turing': self._turing
        }
        func = switcher.get(self._arch, lambda: [])
        return func(a, b, c)
    
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

            source = HMMA_INTS[0][1:]
            debug_print('Pointers1:\n {}\n'.format(pointer))
            debug_print('Source1:\n {}\n'.format(source))
            self._register_files[thread_id].fill(
                a, b, c, pointer, source, enable_c
            )

            pointer[0][0] += 4
            pointer[1][1] += 4
            pointer[2][0] += 4

            source = HMMA_INTS[1][1:]
            debug_print('Source2:\n {}\n'.format(source))
            self._register_files[thread_id].fill(
                a, b, c, pointer, source, enable_c
            )

            pointer[0][0] += 4
            pointer[1][1] += 4

            enable_c = False
            source = HMMA_INTS[2][1:]
            debug_print('Pointers3:\n {}\n'.format(pointer))
            debug_print('Source3:\n {}\n'.format(source))
            self._register_files[thread_id].fill(
                a, b, c, pointer, source, enable_c
            )

            pointer[0][0] += 4
            pointer[1][1] += 4


            source = HMMA_INTS[3][1:]
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
        debug_print('Instruction sources\n{}'.format(instruction_sources))
        for i in range(4):
            if instruction_number % 2 == 0:
                values = self._register_files[thread_group * 4 + i].rf_read(instruction_sources[1])
                address = 'a_{}0'.format(i)
                debug_print('Values:\n1:{}\n2:{}\n'.format(values[0], values[1]))
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
                debug_print('Values:\n1:\n{}2:\n{}'.format(values[0], values[1]))
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
                debug_print('Values:\n1:\n{}2:\n{}'.format(values[0], values[1]))
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
                debug_print('Values:\n1:\n{}2:\n{}'.format(values[0], values[1]))
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
                debug_print('Values:\n1:\n{}2:\n{}'.format(values[0], values[1]))
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
                debug_print('Values:\n1:\n{}2:\n{}'.format(values[0], values[1]))
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
                debug_print('Values:\n1:\n{}2:\n{}'.format(values[0], values[1]))
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
                debug_print('Values:\n1:\n{}2:\n{}'.format(values[0], values[1]))
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
    
    def _volta(self, a, b, c):
        pass
    
    def _pascal(self, a, b, c):
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
    
    def _turing(self, a,  b,  c, n):
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
        sum = 0.0
        for i in range(len(a)):
            sum += a[i] * b[i]
        return sum + c

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
d_real = np.matmul(a, b)
print(d)
print(d_real)