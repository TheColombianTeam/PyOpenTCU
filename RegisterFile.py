import numpy as np


from .common import debug_print


class RegisterFile():
    def __init__(self):
        self._mem = [0] * 24
    
    def rf_write(
            self,
            address,
            value
        ):
        debug_print('Address: {}\nValue:{}'.format(address, value))
        self._mem[address] = value

    def rf_read(self, address):
        return self._mem[address]
    
    def get_addresses(self):
        addresses = [key for key in self._mem]
        return addresses
    
    def fill(self, a, b, c, pointer, source, enable_c):
        for i in range(2):
            value1 = a[pointer[0][1]][pointer[0][0] + i * 2]
            value2 = a[pointer[0][1]][pointer[0][0] + i * 2 + 1]
            self.rf_write(source[0] + i, [value1, value2])

            value1 = b[pointer[1][1] + i * 2][pointer[1][0]]
            value2 = b[pointer[1][1] + i * 2 + 1][pointer[1][0]]
            self.rf_write(source[1] + i, [value1, value2])

            if enable_c:
                value1 = c[pointer[2][1]][pointer[2][0] + i * 2]
                value2 = c[pointer[2][1]][pointer[2][0] + i * 2 + 1]
                self.rf_write(source[2] + i, [value1, value2])
    
    def store(self, filename) -> None:
        file = open(filename, 'w')
        file.write(self.__str__())
        file.close()

    def __str__(self):
        info = '***********************************\n'
        for address, data in enumerate(self._mem):
            if data:
                info += '{}: {}\n'.format(
                    address, 
                    data
                )
        info += '***********************************\n'
        return info
