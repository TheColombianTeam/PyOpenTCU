import numpy as np

from common import debug_print

class TensorBuffer():
    def __init__(self):
        self._buffer = {
            'A0': {},
            'A1': {},
            'B0': {},
            'B1': {},
            'C0': {},
            'C1': {},
            'CX0': {},
            'CX1': {}
        }
    
    def buffer_write(
            self,
            buffer,
            address,
            value,
            pointer
        ):
        key = '{}{}'.format(buffer, pointer)
        debug_print(f"Key {key} address {address} value {value}")
        self._buffer[key][address] = value

    def read_buffer(
            self,
            buffer,
            address,
            pointer
        ):
        key = '{}{}'.format(buffer, pointer)
        return self._buffer[key][address]

    def store(self, filename) -> None:
        file = open(filename, 'w')
        file.write(self.__str__())
        file.close()
    
    def __str__(self):
        info = '***********************************\n'
        for buffer in self._buffer:
            info += '{} buffer:\n'.format(buffer)
            for address in self._buffer[buffer]:
                info += '{} address: {}\n'.format(
                    address, 
                    self._buffer[buffer][address]
                )
            info += '***********************************\n'
        return info
