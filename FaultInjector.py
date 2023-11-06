from sfpy import *


from config import config
from common import debug_print


"""
The fault is composed of:
    target: input(0), output(1), interconnection(2), buffer inputs (3), buffer outputs (4)
    target_thread_group max 8
    element
    mask
    type: bf, sa0, sa1
"""


FAULTTARGET = {
    "INPUT": "0",
    "OUTPUT": "1",
    "INTERCONECCTIONS": "2",
    "BUFFER_INPUTS": "3",
    "BUFFER_OUTPUTS": "4",
}


FAULTTYPE = {"BF": "bf", "SA0": "sa0", "SA1": "sa1"}


class FaultInjector:
    def __init__(self) -> None:
        config_parameters = config()["DEFAULT"]
        self._fault_enable = False
        self._fault_id = None
        self._fault_file_path: str = config_parameters["fault_file"]
        self._type_data: str = config_parameters["type"].lower()
        self._fault = None

    def enable_fault(self, id):
        self._fault_enable = True
        self._fault_id = id
        self._read_fault()

    def _read_fault(self):
        if self._fault_enable:
            file = open(self._fault_file_path, "r")
            self._fault = file.readlines()
            file.close()
            self._fault = self._fault[self._fault_id].split(",")

    def input_fault_inject(self, a, b, c, thread_group):
        if self._fault_enable and self._fault[0] == FAULTTARGET["INPUT"]:
            if self._fault[1] == str(thread_group):
                position = self._fault[2].split("-")
                if position[0].lower() == "a":
                    value = a[int(position[1])][int(position[2])]
                    value = self._inject_fault(value)
                    debug_print(
                        "A[{}][{}]: Expected Value {}, Injected value {}".format(
                            position[1],
                            position[2],
                            a[int(position[1])][int(position[2])],
                            value,
                        )
                    )
                    a[int(position[1])][int(position[2])] = value
                elif position[0].lower() == "b":
                    value = b[int(position[1])][int(position[2])]
                    value = self._inject_fault(value)
                    debug_print(
                        "B[{}][{}]: Expected Value {}, Injected value {}".format(
                            position[1],
                            position[2],
                            b[int(position[1])][int(position[2])],
                            value,
                        )
                    )
                    b[int(position[1])][int(position[2])] = value
                elif position[0].lower() == "c":
                    value = c[int(position[1])]
                    value = self._inject_fault(value)
                    debug_print(
                        "C[{}]: Expected Value {}, Injected value {}".format(
                            position[1], c[int(position[1])], value
                        )
                    )
                    c[int(position[1])] = value
        self._internal_values(a, "A", 16, 4)
        self._internal_values(b, "B", 16, 4)
        self._internal_values(c, "C", 16)
        return a, b, c

    def interconnection_fault_inject(
        self, value, thread_group, interconnection_x, interconnection_y, acc
    ):
        if self._fault_enable and self._fault[0] == FAULTTARGET["INTERCONECCTIONS"]:
            if self._fault[1] == str(thread_group):
                fault = self._fault[2].split("-")
                if (
                    fault[0] == str(interconnection_x)
                    and fault[1] == str(interconnection_y)
                    and fault[2] == str(acc)
                ):
                    value_injected = self._inject_fault(value)
                    debug_print(
                        "Interconnection[{},{},{}]: Expected Value {}, Injected value {}".format(
                            interconnection_x,
                            interconnection_y,
                            acc,
                            value,
                            value_injected,
                        )
                    )
                    return value_injected
        return value

    def output_fault_inject(self, w, thread_group):
        if self._fault_enable and self._fault[0] == FAULTTARGET["OUTPUT"]:
            if self._fault[1] == str(thread_group):
                position = self._fault[2].split("-")
                value = w[int(position[0])][int(position[1])]
                value = self._inject_fault(value)
                debug_print(
                    "W[{}][{}]: Expected Value {}, Injected value {}".format(
                        position[0],
                        position[1],
                        w[int(position[0])][int(position[1])],
                        value,
                    )
                )
                w[int(position[0])][int(position[1])] = value
        self._internal_values(w, "W", 4, 4)
        return w

    def input_buffers(self, value, fault_id):
        if self._fault_enable and self._fault[0] == FAULTTARGET["BUFFER_INPUTS"]:
            if self._fault[2] == fault_id:
                value = self._inject_fault(value)
        return value

    def output_buffers(self, value, fault_id):
        if self._fault_enable and self._fault[0] == FAULTTARGET["BUFFER_OUTPUTS"]:
            if self._fault[2] == fault_id:
                value = self._inject_fault(value)
        return value

    def _inject_fault(self, value):
        if self._type_data == "float16":
            input_format = Float16(value)
        else:
            input_format = Posit16(value)
        input_bits = input_format.bits
        mask = input_format.from_bits(int(self._fault[3], 16))
        mask_bits = mask.bits
        if FAULTTYPE["BF"] in self._fault[4]:
            output = input_bits ^ mask_bits
        elif FAULTTYPE["SA0"] in self._fault[4]:
            mask_bits = ~mask_bits
            output = input_bits & mask_bits
        elif FAULTTYPE["SA1"] in self._fault[4]:
            output = input_bits | mask_bits
        result = input_format.from_bits(output)
        return result

    def _internal_values(self, a, name, rows, columns=0):
        for row in range(rows):
            if not columns == 0:
                for column in range(columns):
                    value = a[row][column]
                    if self._type_data == "float16":
                        input_format = Float16(value)
                    else:
                        input_format = Posit16(value)
                    value_bits = input_format.bits
                    value_hex = hex(value_bits)
            else:
                value = a[row]
                if self._type_data == "float16":
                    input_format = Float16(value)
                else:
                    input_format = Posit16(value)
                value_bits = input_format.bits
                value_hex = hex(value_bits)
