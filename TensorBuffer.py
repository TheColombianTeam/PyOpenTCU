import numpy as np


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
        id = '0' if pointer == 0 else '1'
        key = '{}{}'.format(buffer, id)
        self._buffer[key][address] = value

    def read_buffer(
            self,
            buffer,
            address,
            pointer
        ):
        id = '0' if pointer == 0 else '1'
        key = '{}{}'.format(buffer, id)
        if not address in self._buffer[key]:
            raise Exception(
                'Register file location not initializated'
            )
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
