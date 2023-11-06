from config import config
from sfpy import *
from FaultInjector import FaultInjector


class TensorBuffer:
    def __init__(self, id=0):
        config_parameters = config()["DEFAULT"]
        self._type: str = config_parameters["type"].lower()
        self.__id = id
        self._buffer = {
            "A0": Bank(data_width=16, key="{}{}".format(self.__id, "A0")),
            "A1": Bank(data_width=16, key="{}{}".format(self.__id, "A1")),
            "B0": Bank(data_width=16, key="{}{}".format(self.__id, "B0")),
            "B1": Bank(data_width=16, key="{}{}".format(self.__id, "B1")),
            "C0": Bank(data_width=16, key="{}{}".format(self.__id, "C0")),
            "C1": Bank(data_width=16, key="{}{}".format(self.__id, "C1")),
            "CX0": Bank(data_width=16, key="{}{}".format(self.__id, "CX0")),
            "CX1": Bank(data_width=16, key="{}{}".format(self.__id, "CX1")),
        }

    def buffer_write(self, buffer, address, value, pointer):
        key = "{}{}".format(buffer, pointer)
        value = self.__convert_to_hexa(value)
        address = address.split("_")[-1]
        self._buffer[key].write(address, value)

    def read_buffer(self, buffer, address, pointer):
        key = "{}{}".format(buffer, pointer)
        address = address.split("_")[-1]
        read = self._buffer[key].read(address)
        return self.__convert_from_hexa(read)

    def store(self, filename) -> None:
        file = open(filename, "w")
        file.write(self.__str__())
        file.close()

    def __convert_to_hexa(self, value):
        if self._type == "float16":
            input_format = Float16(value)
        else:
            input_format = Posit16(value)
        input_bits = input_format.bits
        hexa_input = hex(input_bits)
        hexa_input = hexa_input.split("x")[-1]
        return hexa_input

    def __convert_from_hexa(self, value):
        if self._type == "float16":
            input_format = Float16(0.0)
        else:
            input_format = Posit16(0.0)
        value = "0x{}".format(value)
        value = input_format.from_bits(int(value, 16))
        return value

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
    def __init__(self, data_width=32, memory_size=512, key="A0"):
        self.__key = key
        self.__memory_size = memory_size
        self.__registers = [
            Register("{}{}".format(self.__key, id))
            for id in range(self.__memory_size // 128)
        ]
        self.__data_width = data_width

    def write(self, address, value):
        register, cell, num_cells = self.__convert_address(address)
        for idx in range(num_cells):
            substr = 4 * idx
            self.__registers[register].write(cell + idx, value[idx * 4 : substr + 4])

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
        register = cell_location // 8
        cell = cell_location % 8
        if register > self.__memory_size // 128 or cell > 8:
            raise Exception("Error using the memory element")
        return register, cell, num_cells

    def __str__(self):
        string = ""
        for idx, register in enumerate(self.__registers):
            string += "Buffer {}:\nRegister:\n{}".format(idx, register)
        return string


class Register:
    def __init__(self, id="0"):
        self.__id = id
        self.__cells = [Cell("{}{}".format(self.__id, id)) for id in range(128 // 16)]

    def write(self, address, value):
        self.__cells[address].write(value)

    def read(self, address):
        return self.__cells[address].read()

    def __str__(self):
        string = ""
        for idx, value in enumerate(self.__cells):
            string += "Cell {}: {}\n".format(idx, value)
        return string


class Cell(FaultInjector):
    def __init__(self, id="0"):
        super().__init__()
        self.__id = id
        print("Cell {}".format(self.__id))
        self.__value = [None]

    def write(self, value):
        value = self.__convert_from_hexa(value)
        value = self.input_buffers(value, self.__id)
        value = self.__convert_to_hexa(value)
        self.__value[0] = value

    def read(self):
        value = self.__value[0]
        value = self.__convert_from_hexa(value)
        value = self.output_buffers(value, self.__id)
        value = self.__convert_to_hexa(value)
        return value

    def __str__(self):
        return str(self.__value[0])

    def __convert_to_hexa(self, value):
        if self._type == "float16":
            input_format = Float16(value)
        else:
            input_format = Posit16(value)
        input_bits = input_format.bits
        hexa_input = hex(input_bits)
        hexa_input = hexa_input.split("x")[-1]
        return hexa_input

    def __convert_from_hexa(self, value):
        if self._type == "float16":
            input_format = Float16(0.0)
        else:
            input_format = Posit16(0.0)
        value = "0x{}".format(value)
        value = input_format.from_bits(int(value, 16))
        return value
