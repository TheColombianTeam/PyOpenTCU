import numpy as np


from FaultInjector import FAULTTARGET, FAULTTYPE


TOTAL_THREAD_GROUP = 8
INPUTS = ['A', 'B', 'C']


def save_file(faults, filename):
    with open(filename, 'w') as file:
        file.writelines(faults)


def exhaustive():
    faults = []
    targets = FAULTTARGET.keys()
    for target in targets:
        if target == 'BUFFER_INPUTS' or target == 'BUFFER_OUTPUTS':
            thread_group = 0
            for input in INPUTS:
                for bit in range(16):
                    for buffer in range(4):
                        for row in range(4):
                            for column in range(4):
                                element = '{}{}{}{}'.format(input, buffer, row, column)
                                for type in FAULTTYPE.keys():
                                    fault_temp = '{},{},{},{},{}\n'.format(
                                    FAULTTARGET[target],
                                    thread_group,
                                    element,
                                    hex(2 ** bit),
                                    FAULTTYPE[type]
                                    )
                                    if not fault_temp in faults:
                                        faults.append(fault_temp)
        else:
            for thread_group in range(TOTAL_THREAD_GROUP):
                if target == 'INPUT':
                    for input in INPUTS:
                        for row in range(16):
                            for column in range(4):
                                for bit in range(16):
                                    for type in FAULTTYPE.keys():
                                        fault_temp = '{},{},{},{},{}\n'.format(
                                            FAULTTARGET[target],
                                            thread_group,
                                            '{}-{}-{}'.format(input, row, column) if input != 'C' else '{}-{}'.format(input, row),
                                            hex(2 ** bit),
                                            FAULTTYPE[type]
                                        )
                                        if not fault_temp in faults:
                                            faults.append(fault_temp)
                elif target == 'OUTPUT':
                    for row in range(4):
                        for column in range(4):
                            for bit in range(16):
                                for type in FAULTTYPE.keys():
                                    fault_temp = '{},{},{},{},{}\n'.format(
                                        FAULTTARGET[target],
                                        thread_group,
                                        '{}-{}'.format(row, column),
                                        hex(2 ** bit),
                                        FAULTTYPE[type]
                                    )
                                    if not fault_temp in faults:
                                        faults.append(fault_temp)
                else:
                    for row in range(4):
                        for column in range(4):
                            for interconnection in range(4):
                                for bit in range(16):
                                    for type in FAULTTYPE.keys():
                                        fault_temp = '{},{},{},{},{}\n'.format(
                                            FAULTTARGET[target],
                                            thread_group,
                                            '{}-{}-{}'.format(row, column, interconnection),
                                            hex(2 ** bit),
                                            FAULTTYPE[type]
                                        )
                                        if not fault_temp in faults:
                                            faults.append(fault_temp)
    return faults


if __name__ == '__main__':
    faults = exhaustive()
    save_file(faults, 'exhaustive_buffers.csv')
