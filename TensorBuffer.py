import numpy as np


class TensorBuffer:
    def __init__(self):
        self._buffer = {
            "A0": {},
            "A1": {},
            "B0": {},
            "B1": {},
            "C0": {},
            "C1": {},
            "CX0": {},
            "CX1": {},
        }

    def buffer_write(self, buffer, address, value, pointer):
        key = "{}{}".format(buffer, pointer)
        print("Buffer {} {} {}".format(key, address, value))
        self._buffer[key][address] = value

    def read_buffer(self, buffer, address, pointer):
        key = "{}{}".format(buffer, pointer)
        return self._buffer[key][address]

    def store(self, filename) -> None:
        file = open(filename, "w")
        file.write(self.__str__())
        file.close()

    def __str__(self):
        info = "***********************************\n"
        for buffer in self._buffer:
            info += "{} buffer:\n".format(buffer)
            for address in self._buffer[buffer]:
                info += "{} address: {}\n".format(
                    address, self._buffer[buffer][address]
                )
            info += "***********************************\n"
        return info


class Bank:
    def __init__(self, data_width=32):
        self.__registers = [Register() for _ in range(512 // 128)]
        self.__data_width = data_width

    def write(self, address, value):
        register, cell, num_cells = self.__convert_address(address)
        for idx in range(num_cells):
            substr = 4 * idx
            self.__registers[register].write(cell + idx, value[idx:substr])

    def read(self, address):
        register, cell, num_cells = self.__convert_address(address)
        value = ""
        for idx in range(num_cells):
            value += self.__registers[register].read(cell + idx)
        return value

    def __convert_address(self, address):
        address_0, address_1 = int(address[0]), int(address[1])
        element = address_0 * 4 + address_1
        num_cells = self.__data_width // 16
        cell_location = element * num_cells
        register = cell_location // num_cells
        cell = cell_location % num_cells
        if register > 3 or cell > 8:
            raise Exception("Error using the memory element")
        return register, cell, num_cells


class Register:
    def __init__(self):
        self.__cells = [Cell() for _ in range(128 // 16)]

    def write(self, address, value):
        self.__cells[address].write(value)

    def read(self, address):
        return self.__cells[address].read()


class Cell:
    def __init__(self):
        self.__value = [None]

    def write(self, value):
        self.__value[0] = value

    def read(self):
        return self.__value[0]
