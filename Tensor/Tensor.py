import os
from typing import List


from Tensor.hw import HMMA_INTS, SOURCE_INTS
from utils.args import args
from utils.formats import CustomData


from Tensor import Buffer, RegisterFile, TCU


class TensorCore(object):
    def __init__(self):
        self.__arch: str = args.tensor.arch.lower()
        self.__threads_per_warp: int = args.tensor.threads_per_warp
        self.__num_tcus: int = args.tensor.num_tcu
        self.__num_buffers: int = args.tensor.num_buffers
        self.__num_thread_groups: int = args.tensor.num_thread_groups
        self.__tcus: List[TCU] = []
        self.__buffers: List[Buffer] = []
        self.__output_buffers: List[Buffer] = []
        self.__RFs: List[RegisterFile] = []
        self.__a: List[List[CustomData]] = []
        self.__b: List[List[CustomData]] = []
        self.__c: List[CustomData] = []
        self.__d: List[List[CustomData]] = []
        self.__build()

    def mul(self):
        self.__validate_operator()
        self.__fill_RFs()
        self.__execution()


    def __execution(self):
        switcher = {
            'volta': self.__volta
        }
        func = switcher.get(self.__arch)
        return func()

    def __volta(self):
        for idx_inst, inst in enumerate(HMMA_INTS):
            # TODO: Parallelize using the number of TCU available
            # In the default configuration -> 2 TCUs
            # Thread group 0, 1, 4, 5 -> TCU 0
            # Thread group 2, 3, 6, 7 -> TCU 1
            for thread_group in range(self.__num_thread_groups):
                self.__execution(thread_group, idx_inst, inst)
                self.__fill_tensor_buffer(thread_group, inst, idx_inst)

                pointer_tensor_buffer = thread_group % 4
                A = [ [ None ] * 4 ] * 16
                B = [ [ None ] * 4 ] * 16
                C = [ None ] * 16
                D = [ [ None ] * 4 ] * 4
                
                odd_inst = idx_inst % 2

                pointer_a = 1 if not odd_inst and thread_group >= 4 else 0
                pointer_b = 0 if odd_inst else 1
                pointer_c = 0 if thread_group < 4 else 1

                for k in range(4):
                    for i in range(4):
                        for j in range(4):
                            A[k * 4 + i][j] = self.__buffers[pointer_tensor_buffer].read_buffer(
                                'A',
                                'a_{}{}'.format(k, j),
                                pointer_a
                            )
                            B[k * 4 + i][j] = self.__buffers[pointer_tensor_buffer].read_buffer(
                                'B',
                                'b_{}{}'.format(j, i),
                                pointer_b
                            )
                        C[k * 4 + i] = self.__buffers[pointer_tensor_buffer].read_buffer(
                            'C',
                            'c_{}{}'.format(k, i),
                            pointer_c
                        )
                
                for k in range(4):
                    for i in range(4):
                        # TODO: use the TCU unit
                        D[k][i] = self._dot_product(
                            A[k * 4 + i],
                            B[k * 4 + i],
                            C[k * 4 + i],
                            thread_group,
                            k, i
                        )

                buffer_c_des = 0 if thread_group < 4 else 1
                buffer_c_group = 'C' if odd_inst else 'CX'

                for i in range(4):
                    for j in range(4):
                        self.__buffers[pointer_tensor_buffer].buffer_write(
                            buffer_c_group,
                            'c_{}{}'.format(i, j),
                            D[i][j],
                            buffer_c_des
                        )
                
                for i in range(4):
                    for j in range(2):
                        data = [D[i][j * 2], D[i][j * 2 + 1]]
                        self.__RFs[thread_group * 4 + i].rf_write(
                            inst[0] + j,
                            data
                        )

    def __validate_operator(self):
        if len(self.__a) == 0 or len(self.__b) == 0 or len(self.__c) == 0:
            raise Exception('The operands (A, B, and C) must be initializated')
        
    def __fill_tensor_buffer(
            self,
            thread_group: int,
            inst_sources: List[int],
            inst: int
        ):
        pointer_storage = 0 if thread_group < 4 else 0
        pointer_tensor_buffer = thread_group % 4

        for op in range(4):
            if inst % 2:
                for i in range(1, 4):
                    addres_ = 'a' if i == 1 else 'b' if i == 2 else 'c'
                    for y in range(2):
                        values = self.__RFs[thread_group * 4 + op].rf_read(
                            inst_sources[i] + y
                        )

                        for z in range(2):
                            address = '{}_{}{}'.format(addres_, op, y * 2 + z)

                            self.__buffers[pointer_tensor_buffer].buffer_write(
                                addres_.upper(),
                                address,
                                values[z],
                                pointer_storage
                            )
            else:
                for i in range(2):
                    values = self.__RFs[thread_group * 4 + op].rf_read(
                        inst_sources[3] + i
                    )

                    for y in range(2):
                        address = 'c_{}{}'.format(op, i * 2 + y)

                        self.__buffers[pointer_tensor_buffer].buffer_write(
                            'C',
                            address,
                            values[y],
                            pointer_storage
                        )
  
    def __build(self):
        switcher = {
            'volta': self.__volta
        }
        func = switcher.get(self.__arch)
        if not func:
            raise NotImplemented(
                'The {} arch has not been implemented'.format(self.__arch)
            )
        for i in range(self.__num_tcus):
            self.__tcus.append(TCU(i))
        for _ in range(self.__num_buffers):
            self.__buffers.append(Buffer())
            self.__output_buffers.append(Buffer())
        for _ in range(self.__threads_per_warp):
            self.__RFs.append(RegisterFile())

    def __fill_RFs(self):
        pointer = [
            [0, 0],
            [0, 0],
            [0, 0]
        ]

        pointer_ = [
            [0,  0,  0,  0,   0,   0,   0,   0], # [0][0]
            [0,  4, -8, -4, -12,  -8, -20, -16], # [0][1]
            [0, -4,  0, -4, -12, -16, -12, -16], # [1][0]
            [0,  0,  0,  0,   0,   0,   0,   0], # [1][1]
            [0,  0,  8,  8,   0,   0,   8,   0], # [2][0]
            [0,  4, -8, -4, -12,  -8, -20, -16]  # [2][1]
        ]

        for thread_per_warp in range(self.__threads_per_warp):
            idx = thread_per_warp // 4
            pointer[0][0] = 0
            pointer[0][1] = thread_per_warp + pointer_[1][idx]
            pointer[1][0] = thread_per_warp + pointer_[2][idx]
            pointer[1][1] = 0
            pointer[2][0] = pointer_[4][idx]
            pointer[2][1] = thread_per_warp + pointer_[5][idx]

            for i in range(4):
                source = SOURCE_INTS[i][1:]

                enable_c = not i // 2

                self.__RFs[thread_per_warp].fill(
                    self.__a,
                    self.__b,
                    self.__c,
                    pointer,
                    source,
                    enable_c
                )

                pointer[0][0] += 4
                pointer[1][1] += 4

                if i == 0:
                    pointer[2][0] += 4
    
    @property
    def a(self) -> List[List[CustomData]]:
        return self.__a
    
    @a.setter
    def a(self, a:List[List[CustomData]]):
        assert len(a) == 4, ValueError
        self.__a = a

    @property
    def b(self) -> List[List[CustomData]]:
        return self.__b
    
    @b.setter
    def b(self, b:List[List[CustomData]]):
        assert len(b) == 4, ValueError
        self.__b = b

    @property
    def c(self) -> List[CustomData]:
        return self.__c
    
    @c.setter
    def c(self, c:List[CustomData]):
        assert len(c) == 4, ValueError
        self.__c = c
    
    @property
    def d(self) -> List[List[CustomData]]:
        return self.__d
