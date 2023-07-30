# generator of .VHD sass files for FlexGrip, just copy the file in the TP and Prueba folder and use
# this function generates the principal.c file in order to start the automatic sass generation process:

import subprocess
import sys
import os
import io
import struct
import argparse
from datetime import datetime
from datetime import date
import time
import os.path
from os import path

import numpy as np
import os
import matplotlib.pyplot as plt
import random
from sfpy import *  # library to import the posit format of real numbers
import contextlib


disabled = 0
input_target = 1
output_target = 2
internal_target = 3


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def hex_to_float(hex_value):
    # print("starting")

    x = 0.0
    y = 0.0
    hex_valuei = 0
    nan_val = 0
    try:
        hex_valuei = int(hex_value[0], 16) & 0x7
    except:
        nan_val = 1

    if (
        nan_val == 0
    ):  # it means that there should not be any to calculate, there is an U or something as input.
        hex_sign = int(hex_value[0], 16) & 0x8
        if hex_sign == 0x8:
            # 	print("signo -")
            sign = 1
        else:
            # 	print("signo +")
            sign = 0

        string_hex = str(hex_valuei) + hex_value[1:]
        # print( str(hex_valuei))
        # print( str(string_hex))

        x = struct.unpack("f", struct.pack("i", int(string_hex, 16)))

        if sign == 1:
            y = -x[0]
        else:
            y = x[0]

    else:
        x = struct.unpack("f", struct.pack("i", int("0x7fffffff", 16)))
        y = x[0]

    return y


# defining a different screen (not used in this project but can be helpful in other ones):

# ----------------------------------------------------------------------------------------------------------

# # defining a new space to print (a buffer string)
# # saving the original space for printing:
# old_stdout = sys.stdout
# # defining the new buffer space for print
# sys.stdout = buffer = io.StringIO()
# # things to be "printed" in the buffer

# print("this one is new....")
# #	print(buffer.getvalue())

# # returning the original print space to the system, so the normal printing can continue.
# sys.stdout = old_stdout
# # collecting the values from the buffer
# prints_from_buffer = buffer.getvalue()


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# This functional description does not includes the limitations of the real block to perform the half presicion. To be included.

# the half precision only supports a range between: (+/-) 65,504


def FMA_core(source_a, source_b, source_c):
    destiny_d = (source_a * source_b) + source_c

    return destiny_d


# This functional description of a tensor element only accepts inputs which are list of a given size (n).
# n is the HW limit of the


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def Tensor_element(
    source_a, source_b, source_c, n
):  # souce_a and source_b must be lists, source_c is only a constant.
    destiny_d = 0.0  # dest = [0] * n		# initial definition of the zero list.
    # temp_values:
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    w3 = 0.0

    # additional check condition only to be sure that the operation will be performed the same number of element in size.
    if (len(source_a) <= n) and (len(source_b) <= n):
        # it is possible to perform the execution of this module

        w0 = FMA_core(source_a[0], source_b[0], source_c)
        # 		print("W0:" + str(w0))
        w1 = FMA_core(source_a[1], source_b[1], w0)
        # 		print("W1:" + str(w1))
        w2 = FMA_core(source_a[2], source_b[2], w1)
        # 		print("W2:" + str(w2))
        w3 = FMA_core(source_a[3], source_b[3], w2)
        # 		print("W3:" + str(w3))

        # 		Other option (software faster):
        # 		w3 = source_c + source_a[0] * source_b[0]
        # 		for i in range (1, n):
        # 			w3 = w3 + source_a[i] * source_b[i]

        destiny_d = w3

    else:
        print("error using the tensor element...")

    return destiny_d


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def pascal_execution(D, A, B, C, n):
    # Pascal engine: only 16 FFMAs to perform the operations.
    # These are organized as sets of operations that then are propagated among to obtain the result.
    w00 = 0.0
    w01 = 0.0
    w02 = 0.0
    w03 = 0.0
    w10 = 0.0
    w11 = 0.0
    w12 = 0.0
    w13 = 0.0
    w20 = 0.0
    w21 = 0.0
    w22 = 0.0
    w23 = 0.0
    w30 = 0.0
    w31 = 0.0
    w32 = 0.0
    w33 = 0.0

    # First cycle:
    w00 = FMA_core(A[0][0], B[0][0], C[0][0])
    w01 = FMA_core(A[0][0], B[0][1], C[0][1])
    w02 = FMA_core(A[0][0], B[0][2], C[0][2])
    w03 = FMA_core(A[0][0], B[0][3], C[0][3])

    # Second cycle:
    w10 = FMA_core(A[0][1], B[1][0], w00)
    w11 = FMA_core(A[0][1], B[1][1], w01)
    w12 = FMA_core(A[0][1], B[1][2], w02)
    w13 = FMA_core(A[0][1], B[1][3], w03)

    # third cycle:
    w20 = FMA_core(A[0][2], B[2][0], w10)
    w21 = FMA_core(A[0][2], B[2][1], w11)
    w22 = FMA_core(A[0][2], B[2][2], w12)
    w23 = FMA_core(A[0][2], B[2][3], w13)

    # Fourth cycle:
    w30 = FMA_core(A[0][3], B[3][0], w20)
    w31 = FMA_core(A[0][3], B[3][1], w21)
    w32 = FMA_core(A[0][3], B[3][2], w22)
    w33 = FMA_core(A[0][3], B[3][3], w23)

    D[0][0] = w30
    D[0][1] = w31
    D[0][2] = w32
    D[0][3] = w33

    # Fifth cycle:
    w00 = FMA_core(A[1][0], B[0][0], C[1][0])
    w01 = FMA_core(A[1][0], B[0][1], C[1][1])
    w02 = FMA_core(A[1][0], B[0][2], C[1][2])
    w03 = FMA_core(A[1][0], B[0][3], C[1][3])

    # Sixth cycle:
    w10 = FMA_core(A[1][1], B[1][0], w00)
    w11 = FMA_core(A[1][1], B[1][1], w01)
    w12 = FMA_core(A[1][1], B[1][2], w02)
    w13 = FMA_core(A[1][1], B[1][3], w03)

    # seventh cycle:
    w20 = FMA_core(A[1][2], B[2][0], w10)
    w21 = FMA_core(A[1][2], B[2][1], w11)
    w22 = FMA_core(A[1][2], B[2][2], w12)
    w23 = FMA_core(A[1][2], B[2][3], w13)

    # Eight cycle:
    w30 = FMA_core(A[1][3], B[3][0], w20)
    w31 = FMA_core(A[1][3], B[3][1], w21)
    w32 = FMA_core(A[1][3], B[3][2], w22)
    w33 = FMA_core(A[1][3], B[3][3], w23)

    D[1][0] = w30
    D[1][1] = w31
    D[1][2] = w32
    D[1][3] = w33

    # 9 cycle:
    w00 = FMA_core(A[2][0], B[0][0], C[2][0])
    w01 = FMA_core(A[2][0], B[0][1], C[2][1])
    w02 = FMA_core(A[2][0], B[0][2], C[2][2])
    w03 = FMA_core(A[2][0], B[0][3], C[2][3])

    # 10 cycle:
    w10 = FMA_core(A[2][1], B[1][0], w00)
    w11 = FMA_core(A[2][1], B[1][1], w01)
    w12 = FMA_core(A[2][1], B[1][2], w02)
    w13 = FMA_core(A[2][1], B[1][3], w03)

    # 11 cycle:
    w20 = FMA_core(A[2][2], B[2][0], w10)
    w21 = FMA_core(A[2][2], B[2][1], w11)
    w22 = FMA_core(A[2][2], B[2][2], w12)
    w23 = FMA_core(A[2][2], B[2][3], w13)

    # 12 cycle:
    w30 = FMA_core(A[2][3], B[3][0], w20)
    w31 = FMA_core(A[2][3], B[3][1], w21)
    w32 = FMA_core(A[2][3], B[3][2], w22)
    w33 = FMA_core(A[2][3], B[3][3], w23)

    D[2][0] = w30
    D[2][1] = w31
    D[2][2] = w32
    D[2][3] = w33

    # 13 cycle:
    w00 = FMA_core(A[3][0], B[0][0], C[3][0])
    w01 = FMA_core(A[3][0], B[0][1], C[3][1])
    w02 = FMA_core(A[3][0], B[0][2], C[3][2])
    w03 = FMA_core(A[3][0], B[0][3], C[3][3])

    # 14 cycle:
    w10 = FMA_core(A[3][1], B[1][0], w00)
    w11 = FMA_core(A[3][1], B[1][1], w01)
    w12 = FMA_core(A[3][1], B[1][2], w02)
    w13 = FMA_core(A[3][1], B[1][3], w03)

    # 15 cycle:
    w20 = FMA_core(A[3][2], B[2][0], w10)
    w21 = FMA_core(A[3][2], B[2][1], w11)
    w22 = FMA_core(A[3][2], B[2][2], w12)
    w23 = FMA_core(A[3][2], B[2][3], w13)

    # 16 cycle:
    w30 = FMA_core(A[3][3], B[3][0], w20)
    w31 = FMA_core(A[3][3], B[3][1], w21)
    w32 = FMA_core(A[3][3], B[3][2], w22)
    w33 = FMA_core(A[3][3], B[3][3], w23)

    D[3][0] = w30
    D[3][1] = w31
    D[3][2] = w32
    D[3][3] = w33

    return D


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def turing_execution(D, A, B, C, n):
    # Turing engine: 64 FFMAs to perform the operations simultaneously.

    # One cycle:
    D[0][0] = Tensor_element(A[0], B[0], C[0][0], n)  # Each module contains 4 FFMAs
    D[0][1] = Tensor_element(A[0], B[1], C[0][1], n)
    D[0][2] = Tensor_element(A[0], B[2], C[0][2], n)
    D[0][3] = Tensor_element(A[0], B[3], C[0][3], n)

    D[1][0] = Tensor_element(A[1], B[0], C[1][0], n)
    D[1][1] = Tensor_element(A[1], B[1], C[1][1], n)
    D[1][2] = Tensor_element(A[1], B[2], C[1][2], n)
    D[1][3] = Tensor_element(A[1], B[3], C[1][3], n)

    D[2][0] = Tensor_element(A[2], B[0], C[2][0], n)
    D[2][1] = Tensor_element(A[2], B[1], C[2][1], n)
    D[2][2] = Tensor_element(A[2], B[2], C[2][2], n)
    D[2][3] = Tensor_element(A[2], B[3], C[2][3], n)

    D[3][0] = Tensor_element(A[3], B[0], C[3][0], n)
    D[3][1] = Tensor_element(A[3], B[1], C[3][1], n)
    D[3][2] = Tensor_element(A[3], B[2], C[3][2], n)
    D[3][3] = Tensor_element(A[3], B[3], C[3][3], n)

    # End of the cycle.

    return D


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def turing_execution_adv(output_tensor_buffer, tensor_buffer_list):
    print("Input value")

    print(output_tensor_buffer[0].print_buffers())

    n = 16

    # One cycle:
    A = list()
    B = list()
    A.append(tensor_buffer_list[0].buffer_A_read("a_00"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_01"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_02"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_03"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_00"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_10"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_20"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_30"))

    # Each Tensor_element module contains 4 FFMAs
    output_tensor_buffer[0].buffer_C_write(
        "w_003", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_00"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_00"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_01"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_02"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_03"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_01"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_11"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_21"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_31"))

    output_tensor_buffer[0].buffer_C_write(
        "w_013", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_10"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_00"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_01"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_02"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_03"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_02"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_12"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_22"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_32"))

    output_tensor_buffer[0].buffer_C_write(
        "w_023", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_20"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_00"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_01"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_02"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_03"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_03"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_13"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_23"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_33"))

    output_tensor_buffer[0].buffer_C_write(
        "w_033", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_30"), n)
    )

    # -----

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_10"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_11"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_12"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_13"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_00"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_10"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_20"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_30"))

    output_tensor_buffer[0].buffer_C_write(
        "w_103", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_01"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_10"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_11"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_12"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_13"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_01"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_11"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_21"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_31"))

    output_tensor_buffer[0].buffer_C_write(
        "w_113", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_11"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_10"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_11"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_12"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_13"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_02"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_12"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_22"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_32"))

    output_tensor_buffer[0].buffer_C_write(
        "w_123", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_21"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_10"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_11"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_12"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_13"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_03"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_13"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_23"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_33"))

    output_tensor_buffer[0].buffer_C_write(
        "w_133", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_31"), n)
    )

    # ------

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_20"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_21"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_22"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_23"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_00"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_10"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_20"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_30"))

    output_tensor_buffer[0].buffer_C_write(
        "w_203", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_02"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_20"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_21"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_22"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_23"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_01"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_11"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_21"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_31"))

    output_tensor_buffer[0].buffer_C_write(
        "w_213", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_12"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_20"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_21"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_22"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_23"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_02"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_12"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_22"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_32"))

    output_tensor_buffer[0].buffer_C_write(
        "w_223", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_22"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_20"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_21"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_22"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_23"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_03"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_13"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_23"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_33"))

    output_tensor_buffer[0].buffer_C_write(
        "w_233", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_32"), n)
    )

    # -----

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_30"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_31"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_32"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_33"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_00"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_10"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_20"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_30"))

    output_tensor_buffer[0].buffer_C_write(
        "w_303", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_03"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_30"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_31"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_32"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_33"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_01"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_11"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_21"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_31"))

    output_tensor_buffer[0].buffer_C_write(
        "w_313", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_13"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_30"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_31"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_32"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_33"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_02"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_12"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_22"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_32"))

    output_tensor_buffer[0].buffer_C_write(
        "w_323", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_23"), n)
    )

    A.clear()
    B.clear()

    A.append(tensor_buffer_list[0].buffer_A_read("a_30"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_31"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_32"))
    A.append(tensor_buffer_list[0].buffer_A_read("a_33"))

    B.append(tensor_buffer_list[0].buffer_B_read("b_03"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_13"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_23"))
    B.append(tensor_buffer_list[0].buffer_B_read("b_33"))

    output_tensor_buffer[0].buffer_C_write(
        "w_333", Tensor_element(A, B, tensor_buffer_list[0].buffer_C_read("c_33"), n)
    )

    print("output value")

    print(output_tensor_buffer[0].print_buffers())

    return output_tensor_buffer


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class register_file(object):
    # 	mem = {}

    def __init__(self):
        self.mem = {}
        self.datax = 0

    def RF_write(self, address, value):
        self.mem[address] = value

    def RF_read(self, address):
        if address in self.mem:
            data = self.mem[address]
        else:
            data = hex(0)  # register file location not initialized

        return data

    def print_register_file(self):
        print(self.mem)

    def get_addresses(self):
        local_list = []
        for key in self.mem:
            local_list.append(key)
        return local_list


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# this object represents the tensor buffer employed to feed the tensor core with the load values from the RF:


class tensor_buffer(object):
    # 	mem = {}

    def __init__(self):
        self.A0_buffer = {}  # A0 to be employed as inputs to the tensor core.
        self.B0_buffer = {}  # B0
        self.C0_buffer = {}  # C0
        self.C00_buffer = {}  # C0'
        self.A1_buffer = {}  # A1 to be employed as inputs to the tensor core.
        self.B1_buffer = {}  # B1
        self.C1_buffer = {}  # C1
        self.C11_buffer = {}  # C1'

    def buffer_A_write(self, address, value, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            self.A0_buffer[address] = value
        else:  # it means that thread groups 4, 5, 6 or 7 are writing here
            self.A1_buffer[address] = value

    def buffer_B_write(self, address, value, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            self.B0_buffer[address] = value
        else:  # it means that thread groups 4, 5, 6 or 7 are writing here
            self.B1_buffer[address] = value

    def buffer_C_write(self, address, value, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            self.C0_buffer[address] = value
        else:  # it means that thread groups 4, 5, 6 or 7 are writing here
            self.C1_buffer[address] = value

    def buffer_Cx_write(self, address, value, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            self.C00_buffer[address] = value
        else:  # it means that thread groups 4, 5, 6 or 7 are writing here
            self.C11_buffer[address] = value

    def buffer_A_read(self, address, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            if address in self.A0_buffer:
                data = self.A0_buffer[address]
            else:
                data = hex(0)  # register file location not initialized
        else:
            if address in self.A1_buffer:
                data = self.A1_buffer[address]
            else:
                data = hex(0)  # register file location not initialized

        return data

    def buffer_B_read(self, address, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            if address in self.B0_buffer:
                data = self.B0_buffer[address]
            else:
                data = hex(0)  # register file location not initialized
        else:
            if address in self.B1_buffer:
                data = self.B1_buffer[address]
            else:
                data = hex(0)  # register file location not initialized

        return data

    def buffer_C_read(self, address, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            if address in self.C0_buffer:
                data = self.C0_buffer[address]
            else:
                data = hex(0)  # register file location not initialized

        else:
            if address in self.C1_buffer:
                data = self.C1_buffer[address]
            else:
                data = hex(0)  # register file location not initialized

        return data

    def buffer_Cx_read(self, address, pointer_storage):
        if (
            pointer_storage == 0
        ):  # it means that thread groups 0, 1, 2 or 3 are writing here
            if address in self.C00_buffer:
                data = self.C00_buffer[address]
            else:
                data = hex(0)  # register file location not initialized

        else:
            if address in self.C11_buffer:
                data = self.C11_buffer[address]
            else:
                data = hex(0)  # register file location not initialized

        return data

    def print_buffers(self):
        print("************************************")
        print("A0")
        print(self.A0_buffer)
        print("------------")
        print("A1")
        print(self.A1_buffer)
        print("------------")
        print("B0")
        print(self.B0_buffer)
        print("------------")
        print("B1")
        print(self.B1_buffer)
        print("------------")
        print("C0")
        print(self.C0_buffer)
        print("------------")
        print("C1")
        print(self.C1_buffer)
        print("------------")
        print("C00")
        print(self.C00_buffer)
        print("------------")
        print("C11")
        print(self.C11_buffer)
        print("************************************")

    def store_buffers_in_file(self, name):
        new_file = open(name, "w")

        new_file.write("Buffer A0						Buffer A1 \n")

        for key in self.A0_buffer:
            new_file.write(
                str(key)
                + " "
                + str(self.A0_buffer[key])
                + "						"
                + str(key)
                + " "
                + str(self.A1_buffer[key])
                + "\n"
            )

        new_file.write("Buffer B0						Buffer B1 \n")

        for key in self.B0_buffer:
            new_file.write(
                str(key)
                + " "
                + str(self.B0_buffer[key])
                + "						"
                + str(key)
                + " "
                + str(self.B1_buffer[key])
                + "\n"
            )

        new_file.write("Buffer C0						Buffer C1 \n")

        for key in self.C0_buffer:
            new_file.write(
                str(key)
                + " "
                + str(self.C0_buffer[key])
                + "						"
                + str(key)
                + " "
                + str(self.C1_buffer[key])
                + "\n"
            )

        new_file.write("Buffer C00						Buffer C11 \n")

        for key in self.C00_buffer:
            new_file.write(
                str(key)
                + " "
                + str(self.C00_buffer[key])
                + "						"
                + str(key)
                + " "
                + str(self.C11_buffer[key])
                + "\n"
            )

        new_file.close()
        print("Store completed")


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def store_to_file(matrix, file_name, n):
    new_file = open(file_name, "w")

    for i in range(0, n):
        for j in range(0, n):
            new_file.write(str(matrix[i][j]) + "        ")

        new_file.write("\n")


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# process of locating the different values in the register file according to the locations of the sources in the inst.


def fill_RF(
    FI_format,
    pointer_i_a,
    pointer_j_a,
    pointer_i_b,
    pointer_j_b,
    pointer_i_c,
    pointer_j_c,
    RF_x,
    A,
    B,
    C,
    source_regA,
    source_regB,
    source_regC,
    enable_c,
):
    # Store into the RF one fo the rows or columns of A, B or C.

    # rows in A:
    # 	value1 = np.float16(A[pointer_i_a][0 + pointer_j_a])
    # 	value2 = np.float16(A[pointer_i_a][1 + pointer_j_a])

    if FI_format == "FP16":
        value1 = Float16(A[pointer_j_a][pointer_i_a + 0])
        value2 = Float16(A[pointer_j_a][pointer_i_a + 1])

    elif FI_format == "POS16":
        value1 = Posit16(A[pointer_j_a][pointer_i_a + 0])
        value2 = Posit16(A[pointer_j_a][pointer_i_a + 1])

    RF_x.RF_write(source_regA, [value1, value2])
    print("SourceA: {}\nValue: [{}, {}]".format(source_regA, value1, value2))

    if FI_format == "FP16":
        value11 = Float16(A[pointer_j_a][pointer_i_a + 2])
        value22 = Float16(A[pointer_j_a][pointer_i_a + 3])

    elif FI_format == "POS16":
        value11 = Posit16(A[pointer_j_a][pointer_i_a + 2])
        value22 = Posit16(A[pointer_j_a][pointer_i_a + 3])

    RF_x.RF_write(source_regA + 1, [value11, value22])
    print("SourceA: {}\nValue: [{}, {}]".format(source_regA, value11, value22))

    # 	print("filling register file:")
    # 	print(source_regA, value1, value2)
    # 	print(str(source_regA + 1), value11, value22)

    if FI_format == "FP16":
        # columns in B:
        value111 = Float16(B[0 + pointer_j_b][pointer_i_b])
        value222 = Float16(B[1 + pointer_j_b][pointer_i_b])

    elif FI_format == "POS16":
        # columns in B:
        value111 = Posit16(B[0 + pointer_j_b][pointer_i_b])
        value222 = Posit16(B[1 + pointer_j_b][pointer_i_b])

    RF_x.RF_write(source_regB, [value111, value222])
    print("SourceB: {}\nValue: [{}, {}]".format(source_regB, value111, value222))

    if FI_format == "FP16":
        value1111 = Float16(B[2 + pointer_j_b][pointer_i_b])
        value2222 = Float16(B[3 + pointer_j_b][pointer_i_b])

    elif FI_format == "POS16":
        value1111 = Posit16(B[2 + pointer_j_b][pointer_i_b])
        value2222 = Posit16(B[3 + pointer_j_b][pointer_i_b])

    RF_x.RF_write(source_regB + 1, [value1111, value2222])
    print("SourceB: {}\nValue: [{}, {}]".format(source_regB, value1111, value2222))

    if enable_c == 1:
        if FI_format == "FP16":
            # rows in C:
            value1c = Float16(C[pointer_j_c][pointer_i_c + 0])
            value2c = Float16(C[pointer_j_c][pointer_i_c + 1])

        elif FI_format == "POS16":
            value1c = Posit16(C[pointer_j_c][pointer_i_c + 0])
            value2c = Posit16(C[pointer_j_c][pointer_i_c + 1])

        RF_x.RF_write(source_regC, [value1c, value2c])
        print("SourceC: {}\nValue: [{}, {}]".format(source_regC, value1c, value2c))

        if FI_format == "FP16":
            value1x = Float16(C[pointer_j_c][pointer_i_c + 2])
            value2x = Float16(C[pointer_j_c][pointer_i_c + 3])

        elif FI_format == "POS16":
            value1x = Posit16(C[pointer_j_c][pointer_i_c + 2])
            value2x = Posit16(C[pointer_j_c][pointer_i_c + 3])

        RF_x.RF_write(source_regC + 1, [value1x, value2x])
        print("SourceC: {}\nValue: [{}, {}]".format(source_regC, value1x, value2x))


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def fill_full_RF(
    RF_thread,
    A,
    B,
    C,
    total_threads_per_warp,
    instrutions_source_destiny_lists,
    FI_format,
):
    # 	Filling the RF with the data from the matrix:

    # 	Definition of the pointers to address the RFs:

    pointer_i_a = 0
    pointer_j_a = 0
    pointer_i_b = 0
    pointer_j_b = 0
    pointer_i_c = 0
    pointer_j_c = 0

    thread_id = 0  # Initial thread ID value.

    print("filling the register file on the GPU core")

    for i in range(0, total_threads_per_warp):
        thread_id = i

        if i >= 0 and i < 4:  # thread group 0
            # row 0
            pointer_i_a = 0  # for thread 0, 1, 2, 3
            pointer_j_a = i
            # col 0
            pointer_i_b = i  # for thread 0, 1, 2, 3
            pointer_j_b = 0
            # part 0
            pointer_i_c = 0  # for thread 0, 1, 2, 3
            pointer_j_c = i

        elif i >= 4 and i < 8:  # thread group 1
            # row 2
            pointer_i_a = 0  # for thread 0, 1, 2, 3
            pointer_j_a = i + 4
            # col 0
            pointer_i_b = i - 4  # for thread 0, 1, 2, 3
            pointer_j_b = 0
            # part 4
            pointer_i_c = 0  #
            pointer_j_c = i + 4  #

        elif i >= 8 and i < 12:  # thread group 2
            # row 0
            pointer_i_a = 0  # for thread 0, 1, 2, 3
            pointer_j_a = i - 8
            # col 2
            pointer_i_b = i
            pointer_j_b = 0
            # part 1
            pointer_i_c = 8
            pointer_j_c = i - 8

        elif i >= 12 and i < 16:  # thread group 3
            # row 2
            pointer_i_a = 0  # for thread 0, 1, 2, 3
            pointer_j_a = i - 4
            # col 2
            pointer_i_b = i - 4
            pointer_j_b = 0
            # part 5
            pointer_i_c = 8
            pointer_j_c = i - 4

        elif i >= 16 and i < 20:  # thread group 4
            # row 1
            pointer_i_a = 0  # for thread 0, 1, 2, 3
            pointer_j_a = i - 12
            # col 1
            pointer_i_b = i - 12
            pointer_j_b = 0
            # part 2
            pointer_i_c = 0  # ******************************************************pending to check
            pointer_j_c = (
                i - 12
            )  # ******************************************************pending to check

        elif i >= 20 and i < 24:  # thread group 5
            # row 3
            pointer_i_a = 0  #
            pointer_j_a = i - 8
            # col 1
            pointer_i_b = i - 16  #
            pointer_j_b = 0
            # part 6
            pointer_i_c = 0  # ******************************************************pending to check
            pointer_j_c = (
                i - 8
            )  # ******************************************************pending to check

        elif i >= 24 and i < 28:  # thread group 6
            # row 1
            pointer_i_a = 0
            pointer_j_a = i - 20
            # col 3
            pointer_i_b = i - 12
            pointer_j_b = 0
            # part 3
            pointer_i_c = 8
            pointer_j_c = i - 20

        elif i >= 28 and i < 32:  # thread group 7
            # row 3
            pointer_i_a = 0  # for thread 0, 1, 2, 3
            pointer_j_a = i - 16
            # col 3
            pointer_i_b = i - 16  # please check that start on 8
            pointer_j_b = 0
            # part 7
            pointer_i_c = 8
            pointer_j_c = i - 16

        else:
            print(
                "Error... something happen when addressing the RF and locate the operands, maybe the number of Threads is not correct"
            )

        # 	print("storing data for " + str(thread_id))

        print("PointerA: \n[{}, {}]".format(pointer_j_a, pointer_i_a))
        print("PointerB: \n[{}, {}]".format(pointer_j_b, pointer_i_b))
        print("PointerC: \n[{}, {}]".format(pointer_j_c, pointer_i_c))
        enable_c = 1
        #   HMMA R4, R22, R12, R4 (load instruction)
        fill_RF(
            FI_format,
            pointer_i_a,
            pointer_j_a,
            pointer_i_b,
            pointer_j_b,
            pointer_i_c,
            pointer_j_c,
            RF_thread[thread_id],
            A,
            B,
            C,
            instrutions_source_destiny_lists[1],
            instrutions_source_destiny_lists[2],
            instrutions_source_destiny_lists[3],
            enable_c,
        )
        pointer_i_a = pointer_i_a + 4
        pointer_j_b = pointer_j_b + 4
        pointer_i_c = pointer_i_c + 4
        print("PointerA: \n[{}, {}]".format(pointer_j_a, pointer_i_a))
        print("PointerB: \n[{}, {}]".format(pointer_j_b, pointer_i_b))
        print("PointerC: \n[{}, {}]".format(pointer_j_c, pointer_i_c))

        # 		print(pointer_i_a, pointer_j_b, pointer_i_c, pointer_j_a)

        #   HMMA R4, R16, R14, R4 (load instruction)
        fill_RF(
            FI_format,
            pointer_i_a,
            pointer_j_a,
            pointer_i_b,
            pointer_j_b,
            pointer_i_c,
            pointer_j_c,
            RF_thread[thread_id],
            A,
            B,
            C,
            instrutions_source_destiny_lists[5],
            instrutions_source_destiny_lists[6],
            instrutions_source_destiny_lists[7],
            enable_c,
        )
        pointer_i_a = pointer_i_a + 4
        pointer_j_b = pointer_j_b + 4
        print("PointerA: \n[{}, {}]".format(pointer_j_a, pointer_i_a))
        print("PointerB: \n[{}, {}]".format(pointer_j_b, pointer_i_b))
        print("PointerC: \n[{}, {}]".format(pointer_j_c, pointer_i_c))

        # 		print(pointer_i_a, pointer_j_b, pointer_i_c, pointer_j_a)
        enable_c = 0  # c is ready, not to write again in RF.

        #   HMMA R4, R18, R8, R4 (load instruction)
        fill_RF(
            FI_format,
            pointer_i_a,
            pointer_j_a,
            pointer_i_b,
            pointer_j_b,
            pointer_i_c,
            pointer_j_c,
            RF_thread[thread_id],
            A,
            B,
            C,
            instrutions_source_destiny_lists[9],
            instrutions_source_destiny_lists[10],
            instrutions_source_destiny_lists[11],
            enable_c,
        )
        pointer_i_a = pointer_i_a + 4
        pointer_j_b = pointer_j_b + 4
        print("PointerA: \n[{}, {}]".format(pointer_j_a, pointer_i_a))
        print("PointerB: \n[{}, {}]".format(pointer_j_b, pointer_i_b))
        print("PointerC: \n[{}, {}]".format(pointer_j_c, pointer_i_c))

        # 		print(pointer_i_a, pointer_j_b, pointer_i_c, pointer_j_a)

        #   HMMA R4, R2, R10, R4 (load instruction)
        fill_RF(
            FI_format,
            pointer_i_a,
            pointer_j_a,
            pointer_i_b,
            pointer_j_b,
            pointer_i_c,
            pointer_j_c,
            RF_thread[thread_id],
            A,
            B,
            C,
            instrutions_source_destiny_lists[13],
            instrutions_source_destiny_lists[14],
            instrutions_source_destiny_lists[15],
            enable_c,
        )

    return RF_thread


# end of function for the movement from memory to RFs

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def fill_tensor_buffer(
    tensor_buffer_list, thread_group, RF_thread, instruction_sources, instruction_number
):
    pointer_storage = 0
    if thread_group < 4:
        pointer_storage = 0
    else:
        pointer_storage = 1

    pointer_tensor_buffer = (
        thread_group % 4
    )  # this pointer defines the addressing tensor buffer according to the octets.
    print("Tensor buffer pointer {}".format(pointer_tensor_buffer))

    print("Instruction sources\n{}".format(instruction_sources))
    if instruction_number % 2 == 0:
        for i in range(0, 4):  # this is the size (4) of the thread group.
            # for thread  = 0

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[1]
            )  # Initial test, a bigger loop is required in order to operate all thread groups.

            address = "a_" + str(i) + "0"
            value = values[0]  # The first of the operands
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )
            # 		address = "a_" + str(thread_group % total_thread_groups) + str(thread_group % 4)

            tensor_buffer_list[pointer_tensor_buffer].buffer_A_write(
                address, value, pointer_storage
            )

            address = "a_" + str(i) + "1"
            value = values[1]  # The second of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_A_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[1] + 1
            )  # implicit register from the same thread

            address = "a_" + str(i) + "2"
            value = values[0]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_A_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            address = "a_" + str(i) + "3"
            value = values[1]  # The second of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_A_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            # Four values of the thread are correctly load into the buffer locations.

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[2]
            )  # implicit register from the same thread

            address = "b_0" + str(i)
            value = values[0]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_B_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            address = "b_1" + str(i)
            value = values[1]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_B_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[2] + 1
            )  # implicit register from the same thread

            address = "b_2" + str(i)
            value = values[0]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_B_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            address = "b_3" + str(i)
            value = values[1]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_B_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[3]
            )  # implicit register from the same thread

            address = "c_" + str(i) + "0"
            value = values[0]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            address = "c_" + str(i) + "1"
            value = values[1]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[3] + 1
            )  # implicit register from the same thread

            address = "c_" + str(i) + "2"
            value = values[0]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            address = "c_" + str(i) + "3"
            value = values[1]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

    if instruction_number % 2 == 1:
        for i in range(0, 4):  # this is the size (4) of the thread group.
            # for thread  = 0

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[3]
            )  # implicit register from the same thread

            address = "c_" + str(i) + "0"
            value = values[0]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            address = "c_" + str(i) + "1"
            value = values[1]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            values = RF_thread[thread_group * 4 + i].RF_read(
                instruction_sources[3] + 1
            )  # implicit register from the same thread

            address = "c_" + str(i) + "2"
            value = values[0]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

            address = "c_" + str(i) + "3"
            value = values[1]  # The first of the operands
            tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
                address, value, pointer_storage
            )
            print(
                "Address {}\nValues {}\nPointer {}".format(
                    address, values, pointer_tensor_buffer
                )
            )

    return tensor_buffer_list


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def result_storage(
    thread_group, RF_thread, output_tensor_buffer, instrutions_source_destiny_lists
):
    for thread in range(0, 4):
        RF_thread[thread_group * 4 + thread].RF_write(
            instrutions_source_destiny_lists[0],
            [
                output_tensor_buffer[0].buffer_C_read("w_" + str(thread) + "03"),
                output_tensor_buffer[0].buffer_C_read("w_" + str(thread) + "13"),
            ],
        )
        RF_thread[thread_group * 4 + thread].RF_write(
            instrutions_source_destiny_lists[0] + 1,
            [
                output_tensor_buffer[0].buffer_C_read("w_" + str(thread) + "23"),
                output_tensor_buffer[0].buffer_C_read("w_" + str(thread) + "33"),
            ],
        )

    return RF_thread


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def load_from_file(matrix, file_object, n, FI_format):
    i = 0

    for line in file_object:
        words = line.split()

        for j in range(0, n):
            if FI_format == "FP16":
                matrix[i][j] = Float16(words[j])
            elif FI_format == "POS16":
                matrix[i][j] = Posit16(words[j])

        i = i + 1

    return matrix


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


def extract_instructions(list_of_instructions):
    extracted_instructions = list()

    var1 = 0
    var2 = 0
    var3 = 0
    var4 = 0

    for line in list_of_instructions:
        print(line)

        line = line.replace(",", "")  # Cleaning the information of each instruction.
        line = line.replace(";", "")  # Cleaning the information of each instruction.
        words = line.split()
        print(line)

        if (
            "R" in words[1]
        ):  # This means that the destiny register is here (always empty): D
            words[1] = words[1].replace("R", "")
            var1 = words[1]
        # 			print("Destiny register: ")
        # 			print( str( int(words[1]) ) )

        if "R" in words[2]:  # This means that the destiny register is here: A
            words[2] = words[2].replace("R", "")
            words[2] = words[2].replace(".", " ")
            in_words = words[2].split()
            var2 = in_words[0]
        # 			print("Source A register: ")
        # 			print( str( int(in_words[0]) ) )

        if "R" in words[3]:  # This means that the destiny register is here: B
            words[3] = words[3].replace("R", "")
            words[3] = words[3].replace(".", " ")
            in_words = words[3].split()
            var3 = in_words[0]
        # 			print("Source B register: ")
        # 			print( str( int(in_words[0]) ) )

        if "R" in words[4]:  # This means that the destiny register is here: B
            words[4] = words[4].replace("R", "")
            var4 = words[4]
        # 			print("Source C register: ")
        # 			print( str( int( words[4] ) ) )

        extracted_instructions.append([var1, var2, var3, var4])

    # 	print(extracted_instructions)

    return extracted_instructions


# -------------------------------------------------------------------------------------------------------


def store_register_file(name_file, RF_thread):
    new_file = open(name_file, "w")

    for ii in range(0, 32):
        new_file.write("Thread " + str(ii) + "\n")

        local_list = []
        local_list = RF_thread[
            ii
        ].get_addresses()  # retrieving all the possible address in the register file per an specific thread (ii)
        local_list.sort()
        for index in local_list:
            new_file.write(str(index) + " " + str(RF_thread[ii].RF_read(index)) + "\n")
    new_file.close()


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def store_results_flat(
    name_file,
    RF_thread,
    destiny_register1,
    destiny_register2,
    destiny_register3,
    destiny_register4,
):
    new_file = open(name_file, "w")

    for ii in range(0, 32):
        local_list = []
        local_list = RF_thread[
            ii
        ].get_addresses()  # retrieving all the possible address in the register file per an specific thread (ii)
        local_list.sort()

        for index in local_list:
            if index == destiny_register1:
                new_file.write(
                    str(RF_thread[ii].RF_read(index)[0])
                    + "\n"
                    + str(RF_thread[ii].RF_read(index)[1])
                    + "\n"
                )

            if index == destiny_register2:
                new_file.write(
                    str(RF_thread[ii].RF_read(index)[0])
                    + "\n"
                    + str(RF_thread[ii].RF_read(index)[1])
                    + "\n"
                )

            if index == destiny_register3:
                new_file.write(
                    str(RF_thread[ii].RF_read(index)[0])
                    + "\n"
                    + str(RF_thread[ii].RF_read(index)[1])
                    + "\n"
                )

            if index == destiny_register4:
                new_file.write(
                    str(RF_thread[ii].RF_read(index)[0])
                    + "\n"
                    + str(RF_thread[ii].RF_read(index)[1])
                    + "\n"
                )

    new_file.close()


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def dot_product_unit(
    A,
    B,
    c,
    thread_group,
    enabler,
    target_thread_group,
    mask,
    fault_type,
    target_interconnection,
):
    # A and B are composed of 4 elements each one, but can be increased according to the tensor generation.

    # additional check condition only to be sure that the operation will be performed the same number of element in size.
    if len(A) == len(B):
        # it is possible to perform the execution of this module

        # partial dot products:
        A0B0 = A[0] * B[0]
        A1B1 = A[1] * B[1]
        A2B2 = A[2] * B[2]
        A3B3 = A[3] * B[3]

        # place for the injection of a fault: (only active if the fault type is activated)

        if enabler == internal_target:
            if target_thread_group == thread_group:
                if target_interconnection == 0:
                    A0B0 = inject_fault_float(A0B0, mask, fault_type)

                elif target_interconnection == 1:
                    A1B1 = inject_fault_float(A1B1, mask, fault_type)

                elif target_interconnection == 2:
                    A2B2 = inject_fault_float(A2B2, mask, fault_type)

                elif target_interconnection == 3:
                    A3B3 = inject_fault_float(A3B3, mask, fault_type)

        # addition:
        out_c = c + (A0B0 + A1B1 + A2B2 + A3B3)

    else:
        print("error using the tensor element...")
        out_c = 0x00

    return out_c


# -------------------------------------------------------------------------------------------------------
# The parameters:

# 	enabler
# 	target_thread_group
# 	mask
# 	fault_type


# are used to perform the fault injection inside the tensor core units.
def volta_execution(
    tensor_buffer_list,
    thread_group,
    instruction,
    RF_thread,
    set_of_HMMA_instructions,
    enabler,
    target_thread_group,
    mask,
    fault_type,
    debug_mode,
    target_interconnection,
):
    pointer_tensor_buffer = (
        thread_group % 4
    )  # this pointer defines the addressing tensor buffer according to the octets.
    print("Thread group {}".format(thread_group))
    print("Pointer tensor {}".format(pointer_tensor_buffer))
    print("Instruction {}".format(instruction))

    # ask for the instruction even or odd to select the sources in the thread groups

    if (instruction % 2) == 0 and (thread_group < 4):  # its even instruction 0, 2, 4, 6
        buffer_c_destiny = 0
        buffer_c_group = 0

        # load elements into the dot product units:
        # One cycle:
        # ---------------------------------------------------------------------------
        A1 = list()
        B1 = list()
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C00 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_00", 0)

        # ---------------------------------------------------------------------------
        A2 = list()
        B2 = list()
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C10 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_10", 0)

        # ---------------------------------------------------------------------------
        A3 = list()
        B3 = list()
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C20 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_20", 0)

        # ---------------------------------------------------------------------------

        A4 = list()
        B4 = list()
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C30 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_30", 0)

        # ---------------------------------------------------------------------------

        A5 = list()
        B5 = list()
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C01 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_01", 0)

        # ---------------------------------------------------------------------------

        A6 = list()
        B6 = list()
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C11 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_11", 0)

        # ---------------------------------------------------------------------------

        A7 = list()
        B7 = list()
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C21 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_21", 0)

        # ---------------------------------------------------------------------------

        A8 = list()
        B8 = list()
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C31 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_31", 0)

        # ---------------------------------------------------------------------------

        A9 = list()
        B9 = list()
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C02 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_02", 0)

        # ---------------------------------------------------------------------------

        A10 = list()
        B10 = list()
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C12 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_12", 0)

        # ---------------------------------------------------------------------------

        A11 = list()
        B11 = list()
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C22 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_22", 0)

        # ---------------------------------------------------------------------------

        A12 = list()
        B12 = list()
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C32 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_32", 0)

        # ---------------------------------------------------------------------------

        A13 = list()
        B13 = list()
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C03 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_03", 0)

        # ---------------------------------------------------------------------------

        A14 = list()
        B14 = list()
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C13 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_13", 0)

        # ---------------------------------------------------------------------------

        A15 = list()
        B15 = list()
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C23 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_23", 0)

        # ---------------------------------------------------------------------------

        A16 = list()
        B16 = list()
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C33 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_33", 0)

        # ---------------------------------------------------------------------------

    elif (instruction % 2) == 1 and (
        thread_group < 4
    ):  # 	its an odd instrucion: 1, 3, 5, 7
        buffer_c_destiny = 0  # pointer 0
        buffer_c_group = 1  # Cx

        # load elements into the dot product units:
        # One cycle:
        # ---------------------------------------------------------------------------
        A1 = list()
        B1 = list()
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C00 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_00", 0)

        # ---------------------------------------------------------------------------
        A2 = list()
        B2 = list()
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C10 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_10", 0)

        # ---------------------------------------------------------------------------
        A3 = list()
        B3 = list()
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C20 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_20", 0)

        # ---------------------------------------------------------------------------

        A4 = list()
        B4 = list()
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 0))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 0))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 0))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 0))

        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C30 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_30", 0)

        # ---------------------------------------------------------------------------

        A5 = list()
        B5 = list()
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C01 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_01", 0)

        # ---------------------------------------------------------------------------

        A6 = list()
        B6 = list()
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C11 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_11", 0)

        # ---------------------------------------------------------------------------

        A7 = list()
        B7 = list()
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C21 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_21", 0)

        # ---------------------------------------------------------------------------

        A8 = list()
        B8 = list()
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 0))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 0))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 0))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 0))

        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C31 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_31", 0)

        # ---------------------------------------------------------------------------

        A9 = list()
        B9 = list()
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C02 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_02", 0)

        # ---------------------------------------------------------------------------

        A10 = list()
        B10 = list()
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C12 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_12", 0)

        # ---------------------------------------------------------------------------

        A11 = list()
        B11 = list()
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C22 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_22", 0)

        # ---------------------------------------------------------------------------

        A12 = list()
        B12 = list()
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 0))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 0))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 0))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 0))

        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C32 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_32", 0)

        # ---------------------------------------------------------------------------

        A13 = list()
        B13 = list()
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C03 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_03", 0)

        # ---------------------------------------------------------------------------

        A14 = list()
        B14 = list()
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C13 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_13", 0)

        # ---------------------------------------------------------------------------

        A15 = list()
        B15 = list()
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C23 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_23", 0)

        # ---------------------------------------------------------------------------

        A16 = list()
        B16 = list()
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 0))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 0))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 0))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 0))

        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C33 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_33", 0)

        # ---------------------------------------------------------------------------

    elif (instruction % 2) == 0 and (thread_group >= 4):  # 	its an 0, 2, 4, 6
        buffer_c_destiny = 1  # pointer 0
        buffer_c_group = 0  # C

        # load elements into the dot product units:
        # One cycle:
        # ---------------------------------------------------------------------------
        A1 = list()
        B1 = list()
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C00 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_00", 1)

        # ---------------------------------------------------------------------------
        A2 = list()
        B2 = list()
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C10 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_10", 1)

        # ---------------------------------------------------------------------------
        A3 = list()
        B3 = list()
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C20 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_20", 1)

        # ---------------------------------------------------------------------------

        A4 = list()
        B4 = list()
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C30 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_30", 1)

        # ---------------------------------------------------------------------------

        A5 = list()
        B5 = list()
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C01 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_01", 1)

        # ---------------------------------------------------------------------------

        A6 = list()
        B6 = list()
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C11 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_11", 1)

        # ---------------------------------------------------------------------------

        A7 = list()
        B7 = list()
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C21 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_21", 1)

        # ---------------------------------------------------------------------------

        A8 = list()
        B8 = list()
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C31 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_31", 1)

        # ---------------------------------------------------------------------------

        A9 = list()
        B9 = list()
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C02 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_02", 1)

        # ---------------------------------------------------------------------------

        A10 = list()
        B10 = list()
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C12 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_12", 1)

        # ---------------------------------------------------------------------------

        A11 = list()
        B11 = list()
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C22 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_22", 1)

        # ---------------------------------------------------------------------------

        A12 = list()
        B12 = list()
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C32 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_32", 1)

        # ---------------------------------------------------------------------------

        A13 = list()
        B13 = list()
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 0))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 0))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 0))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 0))

        C03 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_03", 1)

        # ---------------------------------------------------------------------------

        A14 = list()
        B14 = list()
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 0))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 0))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 0))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 0))

        C13 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_13", 1)

        # ---------------------------------------------------------------------------

        A15 = list()
        B15 = list()
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 0))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 0))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 0))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 0))

        C23 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_23", 1)

        # ---------------------------------------------------------------------------

        A16 = list()
        B16 = list()
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 0))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 0))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 0))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 0))

        C33 = tensor_buffer_list[pointer_tensor_buffer].buffer_C_read("c_33", 1)

        # ---------------------------------------------------------------------------

    elif (instruction % 2) == 1 and (thread_group >= 4):  # 	its an 1, 3, 5, 7
        buffer_c_destiny = 1  # pointer 1
        buffer_c_group = 1  # Cx

        # load elements into the dot product units:
        # One cycle:
        # ---------------------------------------------------------------------------
        A1 = list()
        B1 = list()
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B1.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C00 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_00", 1)

        # ---------------------------------------------------------------------------
        A2 = list()
        B2 = list()
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B2.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C10 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_10", 1)

        # ---------------------------------------------------------------------------
        A3 = list()
        B3 = list()
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B3.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C20 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_20", 1)

        # ---------------------------------------------------------------------------

        A4 = list()
        B4 = list()
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_00", 1))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_01", 1))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_02", 1))
        A4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_03", 1))

        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B4.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C30 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_30", 1)

        # ---------------------------------------------------------------------------

        A5 = list()
        B5 = list()
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B5.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C01 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_01", 1)

        # ---------------------------------------------------------------------------

        A6 = list()
        B6 = list()
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B6.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C11 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_11", 1)

        # ---------------------------------------------------------------------------

        A7 = list()
        B7 = list()
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B7.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C21 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_21", 1)

        # ---------------------------------------------------------------------------

        A8 = list()
        B8 = list()
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_10", 1))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_11", 1))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_12", 1))
        A8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_13", 1))

        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B8.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C31 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_31", 1)

        # ---------------------------------------------------------------------------

        A9 = list()
        B9 = list()
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B9.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C02 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_02", 1)

        # ---------------------------------------------------------------------------

        A10 = list()
        B10 = list()
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B10.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C12 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_12", 1)

        # ---------------------------------------------------------------------------

        A11 = list()
        B11 = list()
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B11.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C22 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_22", 1)

        # ---------------------------------------------------------------------------

        A12 = list()
        B12 = list()
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_20", 1))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_21", 1))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_22", 1))
        A12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_23", 1))

        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B12.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C32 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_32", 1)

        # ---------------------------------------------------------------------------

        A13 = list()
        B13 = list()
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_00", 1))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_10", 1))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_20", 1))
        B13.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_30", 1))

        C03 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_03", 1)

        # ---------------------------------------------------------------------------

        A14 = list()
        B14 = list()
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_01", 1))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_11", 1))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_21", 1))
        B14.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_31", 1))

        C13 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_13", 1)

        # ---------------------------------------------------------------------------

        A15 = list()
        B15 = list()
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_02", 1))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_12", 1))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_22", 1))
        B15.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_32", 1))

        C23 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_23", 1)

        # ---------------------------------------------------------------------------

        A16 = list()
        B16 = list()
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_30", 1))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_31", 1))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_32", 1))
        A16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_A_read("a_33", 1))

        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_03", 1))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_13", 1))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_23", 1))
        B16.append(tensor_buffer_list[pointer_tensor_buffer].buffer_B_read("b_33", 1))

        C33 = tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_read("c_33", 1)

    print(
        "A\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
            A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16
        )
    )

    print(
        "B\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
            B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16
        )
    )

    print(
        "C\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
            C00,
            C01,
            C02,
            C03,
            C10,
            C11,
            C12,
            C13,
            C20,
            C21,
            C22,
            C23,
            C30,
            C31,
            C32,
            C33,
        )
    )

    # ---------------------------------------------------------------------------
    # Each Tensor_element module is based on the dot product unit (described in the patent)

    # Execution of the tensor:

    # there are two tensor cores per sub-SM (8 per SM composed of 4 sub-SMs), ech one divided in octets
    # the octets represent the distribution in HW of the used units and the operation of the warps, as:
    # octet 0 = thread_group_0 and thread_group_4 (with the combined data of thre_0 and thre_4 is possible to operate the result)
    # octet 1 = thread_group_1 and thread_group_5
    # octet 2 = thread_group_2 and thread_group_6
    # octet 3 = thread_group_3 and thread_group_7
    # tensor_ID = 0 for thread_group 0, 1, 4 and 5		# missing implementation in the execution of the fault injection, pending to add.
    # tensor_ID = 1 for thread_group 2, 3, 6 and 7

    # It must be noted that Step0 and Step1 use the same HW tensor, but each thread group must process in parallel,
    # so it is different the core of thread_group 0 and thread 4 (it means that each thread group uses 16 dot_product_units)

    # injection of faults on the inputs: A1[0], A2[1], ..., B1[2], B2[3], ..., C00[0], C11[0], etc.
    # injection of faults on the outputs of the tensor operation: w_003, w_013, etc.

    # Generating a vector of the input values for the injector:

    # code for the fault injector tool:
    # ****************************************************************************************************
    # ****************************************************************************************************

    if enabler == input_target:
        (a_vectors, b_vectors, c_vectors) = collecting_input_operands(
            A1,
            A2,
            A3,
            A4,
            A5,
            A6,
            A7,
            A8,
            A9,
            A10,
            A11,
            A12,
            A13,
            A14,
            A15,
            A16,
            B1,
            B2,
            B3,
            B4,
            B5,
            B6,
            B7,
            B8,
            B9,
            B10,
            B11,
            B12,
            B13,
            B14,
            B15,
            B16,
            C00,
            C01,
            C02,
            C03,
            C10,
            C11,
            C12,
            C13,
            C20,
            C21,
            C22,
            C23,
            C30,
            C31,
            C32,
            C33,
        )

        (a_vectors, b_vectors, c_vectors) = inject_fault(
            enabler,
            a_vectors,
            b_vectors,
            c_vectors,
            target_thread_group,
            thread_group,
            mask,
            fault_type,
        )

        (
            A1,
            A2,
            A3,
            A4,
            A5,
            A6,
            A7,
            A8,
            A9,
            A10,
            A11,
            A12,
            A13,
            A14,
            A15,
            A16,
            B1,
            B2,
            B3,
            B4,
            B5,
            B6,
            B7,
            B8,
            B9,
            B10,
            B11,
            B12,
            B13,
            B14,
            B15,
            B16,
            C00,
            C01,
            C02,
            C03,
            C10,
            C11,
            C12,
            C13,
            C20,
            C21,
            C22,
            C23,
            C30,
            C31,
            C32,
            C33,
        ) = retrieving_input_operands(
            a_vectors,
            b_vectors,
            c_vectors,
            A1,
            A2,
            A3,
            A4,
            A5,
            A6,
            A7,
            A8,
            A9,
            A10,
            A11,
            A12,
            A13,
            A14,
            A15,
            A16,
            B1,
            B2,
            B3,
            B4,
            B5,
            B6,
            B7,
            B8,
            B9,
            B10,
            B11,
            B12,
            B13,
            B14,
            B15,
            B16,
            C00,
            C01,
            C02,
            C03,
            C10,
            C11,
            C12,
            C13,
            C20,
            C21,
            C22,
            C23,
            C30,
            C31,
            C32,
            C33,
        )

    # ****************************************************************************************************
    # ****************************************************************************************************

    w_003 = dot_product_unit(
        A1,
        B1,
        C00,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_013 = dot_product_unit(
        A2,
        B2,
        C01,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_023 = dot_product_unit(
        A3,
        B3,
        C02,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_033 = dot_product_unit(
        A4,
        B4,
        C03,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )

    w_103 = dot_product_unit(
        A5,
        B5,
        C10,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_113 = dot_product_unit(
        A6,
        B6,
        C11,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_123 = dot_product_unit(
        A7,
        B7,
        C12,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_133 = dot_product_unit(
        A8,
        B8,
        C13,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )

    w_203 = dot_product_unit(
        A9,
        B9,
        C20,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_213 = dot_product_unit(
        A10,
        B10,
        C21,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_223 = dot_product_unit(
        A11,
        B11,
        C22,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_233 = dot_product_unit(
        A12,
        B12,
        C23,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )

    w_303 = dot_product_unit(
        A13,
        B13,
        C30,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_313 = dot_product_unit(
        A14,
        B14,
        C31,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_323 = dot_product_unit(
        A15,
        B15,
        C32,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )
    w_333 = dot_product_unit(
        A16,
        B16,
        C33,
        thread_group,
        enabler,
        target_thread_group,
        mask,
        fault_type,
        target_interconnection,
    )

    # ****************************************************************************************************
    # ****************************************************************************************************
    # this function only uses the c_vectors to collect the operands for the fault injection

    if enabler == output_target:
        (aa_vectors, bb_vectors, cc_vectors) = collecting_output_operands(
            [0, 0, 0],
            [0, 0, 0],
            w_003,
            w_013,
            w_023,
            w_033,
            w_103,
            w_113,
            w_123,
            w_133,
            w_203,
            w_213,
            w_223,
            w_233,
            w_303,
            w_313,
            w_323,
            w_333,
        )

        (aa_vectors, bb_vectors, cc_vectors) = inject_fault(
            enabler,
            aa_vectors,
            bb_vectors,
            cc_vectors,
            target_thread_group,
            thread_group,
            mask,
            fault_type,
        )

        (
            w_003,
            w_013,
            w_023,
            w_033,
            w_103,
            w_113,
            w_123,
            w_133,
            w_203,
            w_213,
            w_223,
            w_233,
            w_303,
            w_313,
            w_323,
            w_333,
        ) = retrieving_output_operands(
            aa_vectors,
            bb_vectors,
            cc_vectors,
            w_003,
            w_013,
            w_023,
            w_033,
            w_103,
            w_113,
            w_123,
            w_133,
            w_203,
            w_213,
            w_223,
            w_233,
            w_303,
            w_313,
            w_323,
            w_333,
        )

    # ****************************************************************************************************
    # ****************************************************************************************************

    print("values on the tensor units:")

    print(w_003, A1, B1, C00)
    print(w_013, A2, B2, C01)
    print(w_023, A3, B3, C02)
    print(w_033, A4, B4, C03)

    print(w_103, A5, B5, C10)
    print(w_113, A6, B6, C11)
    print(w_123, A7, B7, C12)
    print(w_133, A8, B8, C13)

    print(w_203, A9, B9, C20)
    print(w_213, A10, B10, C21)
    print(w_223, A11, B11, C22)
    print(w_233, A12, B12, C23)

    print(w_303, A13, B13, C30)
    print(w_313, A14, B14, C31)
    print(w_323, A15, B16, C32)
    print(w_333, A16, B16, C33)

    # storing the results for the next instruction or thread group.

    if buffer_c_group == 0:  # C
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_00", w_003, buffer_c_destiny
        )  # the destiny also depends on the thread group and the instruction.
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_01", w_013, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_02", w_023, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_03", w_033, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_10", w_103, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_11", w_113, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_12", w_123, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_13", w_133, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_20", w_203, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_21", w_213, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_22", w_223, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_22", w_233, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_30", w_303, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_31", w_313, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_32", w_323, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_C_write(
            "c_33", w_333, buffer_c_destiny
        )

    elif buffer_c_group == 1:  # Cx
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_00", w_003, buffer_c_destiny
        )  # the destiny also depends on the thread group and the instruction.
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_01", w_013, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_02", w_023, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_03", w_033, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_10", w_103, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_11", w_113, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_12", w_123, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_13", w_133, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_20", w_203, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_21", w_213, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_22", w_223, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_23", w_233, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_30", w_303, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_31", w_313, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_32", w_323, buffer_c_destiny
        )
        tensor_buffer_list[pointer_tensor_buffer].buffer_Cx_write(
            "c_33", w_333, buffer_c_destiny
        )

    else:
        print(
            "There are issues in the operation of the tensor and assignation of the outputs..."
        )

    # storage into the register file

    RF_thread[thread_group * 4 + 0].RF_write(
        set_of_HMMA_instructions[instruction][0], [w_003, w_013]
    )
    RF_thread[thread_group * 4 + 0].RF_write(
        set_of_HMMA_instructions[instruction][0] + 1, [w_023, w_033]
    )

    RF_thread[thread_group * 4 + 1].RF_write(
        set_of_HMMA_instructions[instruction][0], [w_103, w_113]
    )
    RF_thread[thread_group * 4 + 1].RF_write(
        set_of_HMMA_instructions[instruction][0] + 1, [w_123, w_133]
    )

    RF_thread[thread_group * 4 + 2].RF_write(
        set_of_HMMA_instructions[instruction][0], [w_203, w_213]
    )
    RF_thread[thread_group * 4 + 2].RF_write(
        set_of_HMMA_instructions[instruction][0] + 1, [w_223, w_233]
    )

    RF_thread[thread_group * 4 + 3].RF_write(
        set_of_HMMA_instructions[instruction][0], [w_303, w_313]
    )
    RF_thread[thread_group * 4 + 3].RF_write(
        set_of_HMMA_instructions[instruction][0] + 1, [w_323, w_333]
    )

    return (tensor_buffer_list, RF_thread)


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def inject_fault(
    enabler,
    a_vectors,
    b_vectors,
    c_vectors,
    target_thread_group,
    thread_group,
    mask,
    fault_type,
):
    if (
        target_thread_group == thread_group
    ):  # indicates the real HW unit associated to the thread groups execution.
        if (
            enabler == input_target
        ):  # This injection can only be performed on the inputs of the Tensor units.
            size1 = len(a_vectors)
            size2 = len(b_vectors)
            size3 = len(c_vectors)

            # selecting one target input from the complete list:
            target = 1

            if target >= 0 and target < size1:
                input_variable = a_vectors[target]
            elif target >= size1 and target < (size1 + size2):
                input_variable = b_vectors[target - size1]
            elif target >= (size1 + size2) and target <= (size1 + size2 + size3):
                input_variable = c_vectors[target - size1 - size2]

            result = inject_fault_float(input_variable, mask, fault_type)
            print("injecting input fault:")

            print(
                input_variable,
                result,
                bin(input_variable.bits)[2:].zfill(16),
                bin(result.bits)[2:].zfill(16),
            )

            if target >= 0 and target < size1:
                a_vectors[target] = result
            elif target >= size1 and target < (size1 + size2):
                b_vectors[target - size1] = result
            elif target >= (size1 + size2) and target <= (size1 + size2 + size3):
                c_vectors[target - size1 - size2] = result

        elif (
            enabler == output_target
        ):  # This injection can only be performed on the outputs of the Tensor units.
            # selecting one target input from the complete list:
            target = 1

            input_variable = c_vectors[target]

            result = inject_fault_float(input_variable, mask, fault_type)
            print("injecting output fault:")
            print(
                input_variable,
                result,
                bin(input_variable.bits)[2:].zfill(16),
                bin(result.bits)[2:].zfill(16),
            )

            c_vectors[target] = result

        else:
            print("injector dissabled")

    return (a_vectors, b_vectors, c_vectors)


# ----------------------------------------------------------------------------------------------------------

# input_variable: posit variable targeting the injection of a fault effect
# mask:  input mask in hex format representing the fault to inject in the variable
# 		 The internal operation of the mask is an binary XOR with the original variable, so bit-fliping the value value.
#
# enable: enable the functionality of the injector


def inject_fault_float(input_variable, mask, typex):
    local_mask = input_variable.from_bits(int(mask, 16))
    var1 = input_variable.bits
    var2 = local_mask.bits

    # bit-flip:
    if typex == "bf":
        output_results = var1 ^ var2

    elif typex == "sa0":
        var2 = ~var2
        output_results = var1 & var2

    elif typex == "sa1":
        output_results = var1 | var2
    res = input_variable.from_bits(output_results)

    return res


# ----------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def collecting_output_operands(
    a_vectors,
    b_vectors,
    w_003,
    w_013,
    w_023,
    w_033,
    w_103,
    w_113,
    w_123,
    w_133,
    w_203,
    w_213,
    w_223,
    w_233,
    w_303,
    w_313,
    w_323,
    w_333,
):
    c_vectors = [
        w_003,
        w_013,
        w_023,
        w_033,
        w_103,
        w_113,
        w_123,
        w_133,
        w_203,
        w_213,
        w_223,
        w_233,
        w_303,
        w_313,
        w_323,
        w_333,
    ]

    return (a_vectors, b_vectors, c_vectors)


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def retrieving_output_operands(
    a_vectors,
    b_vectors,
    c_vectors,
    w_003,
    w_013,
    w_023,
    w_033,
    w_103,
    w_113,
    w_123,
    w_133,
    w_203,
    w_213,
    w_223,
    w_233,
    w_303,
    w_313,
    w_323,
    w_333,
):
    w_003 = c_vectors[0]
    w_013 = c_vectors[1]
    w_023 = c_vectors[2]
    w_033 = c_vectors[3]
    w_103 = c_vectors[4]
    w_113 = c_vectors[5]
    w_123 = c_vectors[6]
    w_133 = c_vectors[7]
    w_203 = c_vectors[8]
    w_213 = c_vectors[9]
    w_223 = c_vectors[10]
    w_233 = c_vectors[11]
    w_303 = c_vectors[12]
    w_313 = c_vectors[13]
    w_323 = c_vectors[14]
    w_333 = c_vectors[15]

    return (
        w_003,
        w_013,
        w_023,
        w_033,
        w_103,
        w_113,
        w_123,
        w_133,
        w_203,
        w_213,
        w_223,
        w_233,
        w_303,
        w_313,
        w_323,
        w_333,
    )


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def collecting_input_operands(
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    A8,
    A9,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
    A16,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    B10,
    B11,
    B12,
    B13,
    B14,
    B15,
    B16,
    C00,
    C01,
    C02,
    C03,
    C10,
    C11,
    C12,
    C13,
    C20,
    C21,
    C22,
    C23,
    C30,
    C31,
    C32,
    C33,
):
    a_vectors = [
        A1[0],
        A1[1],
        A1[2],
        A1[3],
        A2[0],
        A2[1],
        A2[2],
        A2[3],
        A3[0],
        A3[1],
        A3[2],
        A3[3],
        A4[0],
        A4[1],
        A4[2],
        A4[3],
        A5[0],
        A5[1],
        A5[2],
        A5[3],
        A6[0],
        A6[1],
        A6[2],
        A6[3],
        A7[0],
        A7[1],
        A7[2],
        A7[3],
        A8[0],
        A8[1],
        A8[2],
        A8[3],
        A9[0],
        A9[1],
        A9[2],
        A9[3],
        A10[0],
        A10[1],
        A10[2],
        A10[3],
        A11[0],
        A11[1],
        A11[2],
        A11[3],
        A12[0],
        A12[1],
        A12[2],
        A12[3],
        A13[0],
        A13[1],
        A13[2],
        A13[3],
        A14[0],
        A14[1],
        A14[2],
        A14[3],
        A15[0],
        A15[1],
        A15[2],
        A15[3],
        A16[0],
        A16[1],
        A16[2],
        A16[3],
    ]

    b_vectors = [
        B1[0],
        B1[1],
        B1[2],
        B1[3],
        B2[0],
        B2[1],
        B2[2],
        B2[3],
        B3[0],
        B3[1],
        B3[2],
        B3[3],
        B4[0],
        B4[1],
        B4[2],
        B4[3],
        B5[0],
        B5[1],
        B5[2],
        B5[3],
        B6[0],
        B6[1],
        B6[2],
        B6[3],
        B7[0],
        B7[1],
        B7[2],
        B7[3],
        B8[0],
        B8[1],
        B8[2],
        B8[3],
        B9[0],
        B9[1],
        B9[2],
        B9[3],
        B10[0],
        B10[1],
        B10[2],
        B10[3],
        B11[0],
        B11[1],
        B11[2],
        B11[3],
        B12[0],
        B12[1],
        B12[2],
        B12[3],
        B13[0],
        B13[1],
        B13[2],
        B13[3],
        B14[0],
        B14[1],
        B14[2],
        B14[3],
        B15[0],
        B15[1],
        B15[2],
        B15[3],
        B16[0],
        B16[1],
        B16[2],
        B16[3],
    ]

    c_vectors = [
        C00,
        C01,
        C02,
        C03,
        C10,
        C11,
        C12,
        C13,
        C20,
        C21,
        C22,
        C23,
        C30,
        C31,
        C32,
        C33,
    ]

    return (a_vectors, b_vectors, c_vectors)


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def retrieving_input_operands(
    a_vectors,
    b_vectors,
    c_vectors,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    A8,
    A9,
    A10,
    A11,
    A12,
    A13,
    A14,
    A15,
    A16,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
    B10,
    B11,
    B12,
    B13,
    B14,
    B15,
    B16,
    C00,
    C01,
    C02,
    C03,
    C10,
    C11,
    C12,
    C13,
    C20,
    C21,
    C22,
    C23,
    C30,
    C31,
    C32,
    C33,
):
    A1[0] = a_vectors[0]
    A1[1] = a_vectors[1]
    A1[2] = a_vectors[2]
    A1[3] = a_vectors[3]
    A2[0] = a_vectors[4]
    A2[1] = a_vectors[5]
    A2[2] = a_vectors[6]
    A2[3] = a_vectors[7]
    A3[0] = a_vectors[8]
    A3[1] = a_vectors[9]
    A3[2] = a_vectors[10]
    A3[3] = a_vectors[11]
    A4[0] = a_vectors[12]
    A4[1] = a_vectors[13]
    A4[2] = a_vectors[14]
    A4[3] = a_vectors[15]
    A5[0] = a_vectors[16]
    A5[1] = a_vectors[17]
    A5[2] = a_vectors[18]
    A5[3] = a_vectors[19]
    A6[0] = a_vectors[20]
    A6[1] = a_vectors[21]
    A6[2] = a_vectors[22]
    A6[3] = a_vectors[23]
    A7[0] = a_vectors[24]
    A7[1] = a_vectors[25]
    A7[2] = a_vectors[26]
    A7[3] = a_vectors[27]
    A8[0] = a_vectors[28]
    A8[1] = a_vectors[29]
    A8[2] = a_vectors[30]
    A8[3] = a_vectors[31]
    A9[0] = a_vectors[32]
    A9[1] = a_vectors[33]
    A9[2] = a_vectors[34]
    A9[3] = a_vectors[35]
    A10[0] = a_vectors[36]
    A10[1] = a_vectors[37]
    A10[2] = a_vectors[38]
    A10[3] = a_vectors[39]
    A11[0] = a_vectors[40]
    A11[1] = a_vectors[41]
    A11[2] = a_vectors[42]
    A11[3] = a_vectors[43]
    A12[0] = a_vectors[44]
    A12[1] = a_vectors[45]
    A12[2] = a_vectors[46]
    A12[3] = a_vectors[47]
    A13[0] = a_vectors[48]
    A13[1] = a_vectors[49]
    A13[2] = a_vectors[50]
    A13[3] = a_vectors[51]
    A14[0] = a_vectors[52]
    A14[1] = a_vectors[53]
    A14[2] = a_vectors[54]
    A14[3] = a_vectors[55]
    A15[0] = a_vectors[56]
    A15[1] = a_vectors[57]
    A15[2] = a_vectors[58]
    A15[3] = a_vectors[59]
    A16[0] = a_vectors[60]
    A16[1] = a_vectors[61]
    A16[2] = a_vectors[62]
    A16[3] = a_vectors[63]

    B1[0] = b_vectors[0]
    B1[1] = b_vectors[1]
    B1[2] = b_vectors[2]
    B1[3] = b_vectors[3]
    B2[0] = b_vectors[4]
    B2[1] = b_vectors[5]
    B2[2] = b_vectors[6]
    B2[3] = b_vectors[7]
    B3[0] = b_vectors[8]
    B3[1] = b_vectors[9]
    B3[2] = b_vectors[10]
    B3[3] = b_vectors[11]
    B4[0] = b_vectors[12]
    B4[1] = b_vectors[13]
    B4[2] = b_vectors[14]
    B4[3] = b_vectors[15]
    B5[0] = b_vectors[16]
    B5[1] = b_vectors[17]
    B5[2] = b_vectors[18]
    B5[3] = b_vectors[19]
    B6[0] = b_vectors[20]
    B6[1] = b_vectors[21]
    B6[2] = b_vectors[22]
    B6[3] = b_vectors[23]
    B7[0] = b_vectors[24]
    B7[1] = b_vectors[25]
    B7[2] = b_vectors[26]
    B7[3] = b_vectors[27]
    B8[0] = b_vectors[28]
    B8[1] = b_vectors[29]
    B8[2] = b_vectors[30]
    B8[3] = b_vectors[31]
    B9[0] = b_vectors[32]
    B9[1] = b_vectors[33]
    B9[2] = b_vectors[34]
    B9[3] = b_vectors[35]
    B10[0] = b_vectors[36]
    B10[1] = b_vectors[37]
    B10[2] = b_vectors[38]
    B10[3] = b_vectors[39]
    B11[0] = b_vectors[40]
    B11[1] = b_vectors[41]
    B11[2] = b_vectors[42]
    B11[3] = b_vectors[43]
    B12[0] = b_vectors[44]
    B12[1] = b_vectors[45]
    B12[2] = b_vectors[46]
    B12[3] = b_vectors[47]
    B13[0] = b_vectors[48]
    B13[1] = b_vectors[49]
    B13[2] = b_vectors[50]
    B13[3] = b_vectors[51]
    B14[0] = b_vectors[52]
    B14[1] = b_vectors[53]
    B14[2] = b_vectors[54]
    B14[3] = b_vectors[55]
    B15[0] = b_vectors[56]
    B15[1] = b_vectors[57]
    B15[2] = b_vectors[58]
    B15[3] = b_vectors[59]
    B16[0] = b_vectors[60]
    B16[1] = b_vectors[61]
    B16[2] = b_vectors[62]
    B16[3] = b_vectors[63]

    C00 = c_vectors[0]
    C01 = c_vectors[1]
    C02 = c_vectors[2]
    C03 = c_vectors[3]
    C10 = c_vectors[4]
    C11 = c_vectors[5]
    C12 = c_vectors[6]
    C13 = c_vectors[7]
    C20 = c_vectors[8]
    C21 = c_vectors[9]
    C22 = c_vectors[10]
    C23 = c_vectors[11]
    C30 = c_vectors[12]
    C31 = c_vectors[13]
    C32 = c_vectors[14]
    C33 = c_vectors[15]

    return (
        A1,
        A2,
        A3,
        A4,
        A5,
        A6,
        A7,
        A8,
        A9,
        A10,
        A11,
        A12,
        A13,
        A14,
        A15,
        A16,
        B1,
        B2,
        B3,
        B4,
        B5,
        B6,
        B7,
        B8,
        B9,
        B10,
        B11,
        B12,
        B13,
        B14,
        B15,
        B16,
        C00,
        C01,
        C02,
        C03,
        C10,
        C11,
        C12,
        C13,
        C20,
        C21,
        C22,
        C23,
        C30,
        C31,
        C32,
        C33,
    )


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


def store_register_file(name_file, RF_thread):
    new_file = open(name_file, "w")

    for ii in range(0, 32):
        new_file.write("Thread " + str(ii) + "\n")

        local_list = []
        local_list = RF_thread[
            ii
        ].get_addresses()  # retrieving all the possible address in the register file per an specific thread (ii)
        local_list.sort()
        for index in local_list:
            new_file.write(
                str(index)
                + " [ "
                + str(RF_thread[ii].RF_read(index)[0])
                + ", "
                + str(RF_thread[ii].RF_read(index)[1])
                + " ] \n"
            )

    new_file.close()


# -------------------------------------------------------------------------------------------------------


def store_bin_register_file(name_file, RF_thread):
    new_file = open(name_file, "w")

    for ii in range(0, 32):
        new_file.write("Thread " + str(ii) + "\n")

        local_list = []
        local_list = RF_thread[
            ii
        ].get_addresses()  # retrieving all the possible address in the register file per an specific thread (ii)
        local_list.sort()

        for index in local_list:
            valxxx = RF_thread[ii].RF_read(index)
            valxxx1 = bin(valxxx[0].bits)
            valxxx2 = bin(valxxx[1].bits)

            new_file.write(
                str(index)
                + "  [ "
                + str(valxxx1[2:].zfill(16))
                + ", "
                + str(valxxx2[2:].zfill(16))
                + " ]"
                + "\n"
            )

    new_file.close()


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

# arguments when launching the simulation:
# FI_output_fault_name
# FI_enabler
# FI_target_thread_group
# FI_mask
# FI_fault_type
# target_interconnection
# debug


# 	FI_enabler = [disabled, input_target, output_target, internal_target]
# FI_target_thread_group = (0 - 7)
# 	FI_mask = 0x4000
# 	FI_fault_type = ("sa0", "sa1")
# target_interconnection = location of the port to inject the fault (0 to n), in the internal injection it is (0 to 3)
# debug_mode = 0
# CHANGE THIS VARIABLE IN ORDER TO ACTIVATE THE PRINTING OF THE DETAILED EXECUTION OF THE SIMULATOR:
# (REGISTER FILE LOAD, BUFFER LOAD AND TENSOR PROCESSING)


# 	total_masks = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]

# 	total_masks_32 = [0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040, 0x00000080, 0x00000100, 0x00000200, 0x00000400, 0x00000800, 0x00001000, 0x00002000, 0x00004000, 0x00008000, 0x00010000, 0x00020000, 0x00040000, 0x00080000, 0x00100000, 0x00200000, 0x00400000, 0x00800000, 0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000]

# 	total_masks_16 = [0x0001, 0x0002, 0x0004 , 0x0008, 0x0010, 0x0020, 0x0040 , 0x0080, 0x0100, 0x0200, 0x0400 , 0x0800, 0x1000, 0x2000, 0x4000 , 0x8000]

#   format = [FP16, POS16]


# generalization of the tool, to load fp16 or posit16:


# Commands to launch:
# python3 functional_tensor_sfpy_FP16.py  FI_output_fault_name  format  FI_enabler  FI_target_thread_group  FI_mask  FI_fault_type  target_interconnection  debug


# example of launching the general version: python3 functional_tensor_sfpy_FP16.py output_folder format disabled 0 0x4000 sa0 0 0

# example of launching: python3 functional_tensor_sfpy_FP16.py output/ FP16 disabled 0 0x4000 sa0 0 0


# python functional_tensor_sfpy_general.py output_folder FP16 disabled 0 0x4000 sa0 0 0
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("FI_output_fault_name", help="fault_name", type=str)
    parser.add_argument("FI_format", help="format", type=str)
    parser.add_argument("FI_enabler", help="enabler", type=str)
    parser.add_argument("FI_target_thread_group", help="target_thread_group", type=int)
    parser.add_argument("FI_mask", help="mask", type=str)
    parser.add_argument("FI_fault_type", help="fault_type", type=str)
    parser.add_argument("target_interconnection", help="target_location", type=int)
    parser.add_argument("debug", help="debug", type=str)

    args = parser.parse_args()

    enabler = args.FI_enabler

    if enabler == "disabled":
        FI_enabler = disabled
    elif enabler == "input_target":
        FI_enabler = input_target
    elif enabler == "output_target":
        FI_enabler = output_target
    elif enabler == "internal_target":
        FI_enabler = internal_target

    FI_target_thread_group = args.FI_target_thread_group
    FI_mask_str = args.FI_mask
    FI_mask = hex(int(FI_mask_str, 16))

    FI_format = args.FI_format

    target_interconnection = args.target_interconnection

    print(FI_mask)

    FI_fault_type = args.FI_fault_type
    debug_mode = args.debug
    FI_output_fault_name = args.FI_output_fault_name

    input_folder = "inputs"
    output_folder = FI_output_fault_name  # "results_posit16"

    if not path.exists(output_folder + "/"):
        os.system("mkdir " + output_folder)

    timestamp1 = time.time()

    total_threads_per_warp = 32
    total_tensor_buffers = (
        4  # equal to the total number of tensor cores (dot product units)
    )

    # length of the matrix and HW engine for the operations:
    n = 16

    # ***************************************************************************************************
    # Generation of the RFs for the threads
    RF_thread = list()

    tensor_buffer_list = list()
    output_tensor_buffer = list()

    for i in range(total_threads_per_warp):
        RF_thread.append(register_file())

    for i in range(total_tensor_buffers):
        tensor_buffer_list.append(tensor_buffer())
        output_tensor_buffer.append(tensor_buffer())

    # Generation of the input matrices for the tensor core of (nxn) sizes.

    if FI_format == "FP16":
        A = np.ones((n, n), dtype=Float16)
        B = np.ones((n, n), dtype=Float16)
        C = np.ones((n, n), dtype=Float16)
        D = np.ones((n, n), dtype=Float16)

    elif FI_format == "POS16":
        A = np.ones((n, n), dtype=Posit16)
        B = np.ones((n, n), dtype=Posit16)
        C = np.ones((n, n), dtype=Posit16)
        D = np.ones((n, n), dtype=Posit16)

    else:
        print(
            "There is an error in the selection of the format, check it again and launch"
        )
        exit()

    file_A = open(input_folder + "/A.txt", "r")
    file_B = open(input_folder + "/B.txt", "r")
    file_C = open(input_folder + "/C.txt", "r")

    A = load_from_file(A, file_A, n, FI_format)
    B = load_from_file(B, file_B, n, FI_format)
    C = load_from_file(C, file_C, n, FI_format)

    # 	list_of_instructions = open("sass.txt", "r")

    # definition of the list containing the values of the sources and destinies for each thread. (extracted from the HMMA instructions)

    instrutions_source_destiny_lists = list()

    # Temp load of the values, the function must load them from the original HMMA instructions.
    instrutions_source_destiny_lists.append(4)  # D
    instrutions_source_destiny_lists.append(22)  # A
    instrutions_source_destiny_lists.append(12)  # B
    instrutions_source_destiny_lists.append(4)  # C

    instrutions_source_destiny_lists.append(6)  # D
    instrutions_source_destiny_lists.append(16)  # A
    instrutions_source_destiny_lists.append(14)  # B
    instrutions_source_destiny_lists.append(6)  # C

    instrutions_source_destiny_lists.append(4)  # D
    instrutions_source_destiny_lists.append(18)  # A
    instrutions_source_destiny_lists.append(8)  # B
    instrutions_source_destiny_lists.append(4)  # C

    instrutions_source_destiny_lists.append(6)  # D
    instrutions_source_destiny_lists.append(2)  # A
    instrutions_source_destiny_lists.append(10)  # B
    instrutions_source_destiny_lists.append(6)  # C

    print("Instructions:\n{}".format(instrutions_source_destiny_lists))
    # Start of function for the movement from memory to RFs

    RF_thread = fill_full_RF(
        RF_thread,
        A,
        B,
        C,
        total_threads_per_warp,
        instrutions_source_destiny_lists,
        FI_format,
    )  # Sending the RF_thread and the input matrices for the filling, retrieving the RF_thread object

    # 	for ii in range (0, 32):
    # 		print("Thread " + str(ii))
    # 		RF_thread[ii].print_register_file()

    name_file = output_folder + "/RF_before_Tensor_execution.txt"
    store_register_file(name_file, RF_thread)

    name_file = output_folder + "/bin_RF_before_Tensor_execution.txt"
    store_bin_register_file(name_file, RF_thread)

    A_golden = A
    B_golden = B
    C_golden = C
    D_golden = C

    print("starting the operation of the tensor: ")

    print("Starting golden operation:")
    D_golden = np.matmul(A_golden, B_golden) + C_golden

    if debug_mode == 1:
        print(np.matmul(A_golden, B_golden) + C_golden)  # golden operation.

    print("Starting golden structural execution:")

    # Warp organization functions:
    # warp_operation() # pending for the moment...

    # it is for now assumed that the input operands are ready A, b, and C.

    total_thread_groups = 8

    # Load the values into the buffers of the octets, so it would be possible to operate independently in the tensor cores.

    #  temporal description of the instructions:

    set_of_HMMA_instructions = [
        [4, 22, 12, 4],
        [6, 22, 12, 6],
        [4, 16, 14, 4],
        [6, 16, 14, 6],
        [4, 18, 8, 4],
        [6, 18, 8, 6],
        [4, 2, 10, 4],
        [6, 2, 10, 6],
    ]

    for instruction in range(0, len(set_of_HMMA_instructions)):
        # 	read_instruction() # this function takes the input ins. and extract the values and place in a kind of buffer to sent into the Tensors.
        # Interconnection and launch of the tensor operation: (use of the instructions to execute each sequence)

        print("======>>> executing instruction: " + str(instruction))

        # 	for warps in task (block)...

        # for thread in range(0, total_threads_per_warp):			# Possibly it must be done by thread group, such as:
        for thread_group in range(0, total_thread_groups):
            # 	if (thread % total_threads_per_warp) >= 0 and (thread % total_threads_per_warp) < 4:		# generalized for N threads in a block.

            # loading the first 16 values for the operation (the threads 0 to 3 perform the operation)

            # print("Thread group:  " + str(thread_group) + " execution")

            # if (instruction == 0) or (instruction == 1):	# case of the first instruction:
            # tasks to perform:
            # 1) fill the tensor buffer with data of each group in octets
            # 2) execute the first tensor operation for all thread groups
            # 3) store results into the tensor buffer

            # replace the instruction source destiny list by the values coming from the extracted instructions.
            tensor_buffer_list = fill_tensor_buffer(
                tensor_buffer_list,
                thread_group,
                RF_thread,
                set_of_HMMA_instructions[instruction],
                instruction,
            )

            print("Original")
            for buffer in tensor_buffer_list:
                print("--------------------------")
                buffer.print_buffers()

            if debug_mode == 1:
                tensor_buffer_list[0].print_buffers()

            (tensor_buffer_list, RF_thread) = volta_execution(
                tensor_buffer_list,
                thread_group,
                instruction,
                RF_thread,
                set_of_HMMA_instructions,
                FI_enabler,
                FI_target_thread_group,
                FI_mask,
                FI_fault_type,
                debug_mode,
                target_interconnection,
            )

            if instruction >= 2:
                if debug_mode == 1:
                    name_file = (
                        output_folder
                        + "/"
                        + "tensor_buffer0_"
                        + str(instruction)
                        + ".txt"
                    )
                    tensor_buffer_list[0].store_buffers_in_file(name_file)

                    name_file = (
                        output_folder
                        + "/"
                        + "tensor_buffer1_"
                        + str(instruction)
                        + ".txt"
                    )
                    tensor_buffer_list[1].store_buffers_in_file(name_file)

                    name_file = (
                        output_folder
                        + "/"
                        + "tensor_buffer2_"
                        + str(instruction)
                        + ".txt"
                    )
                    tensor_buffer_list[2].store_buffers_in_file(name_file)

                    name_file = (
                        output_folder
                        + "/"
                        + "tensor_buffer3_"
                        + str(instruction)
                        + ".txt"
                    )
                    tensor_buffer_list[3].store_buffers_in_file(name_file)

    name_file = output_folder + "/RF_after_Tensor_execution.txt"
    store_register_file(name_file, RF_thread)

    name_file = output_folder + "/bin_RF_after_Tensor_execution.txt"
    store_bin_register_file(name_file, RF_thread)

    store_to_file(D_golden, output_folder + "/D_golden.txt", n)

    timestamp2 = time.time()

    print("Finishing tensor... total execution time:" + str(timestamp2 - timestamp1))

    matrix = np.ones((n, n), dtype=Float16)

    print(set_of_HMMA_instructions[0][0], set_of_HMMA_instructions[1][0])

    name_file = output_folder + "/RF_after_Tensor_execution.txt"

    filex = open(name_file, "r")

    dictionar = dict()

    for line in filex:
        words = line.split(" ")  # Use the space to identify the number of elements:

        if len(words) > 2:  # It means, is the line to be read
            if (
                (int(words[0]) == 4)
                or (int(words[0]) == 5)
                or (int(words[0]) == 6)
                or (int(words[0]) == 7)
            ):
                print("Words: {}".format(words))

                clean = words[2].replace(",", "")
                flat_list.append(clean)
                flat_list.append(words[3])

                if int(words[0]) == 7:
                    dictionar[key] = flat_list
                    print("Flat list {}, Key {}".format(flat_list, key))

        else:
            key = int(words[1])
            flat_list = list()

    print(dictionar)

    for thread_id in range(0, 32):
        if thread_id >= 0 and thread_id < 4:
            for i in range(0, 8):
                matrix[thread_id][i] = dictionar[thread_id][i]

                print(matrix)

        elif thread_id >= 4 and thread_id < 8:
            for i in range(0, 8):
                matrix[thread_id + 4][i] = dictionar[thread_id][i]

        elif thread_id >= 8 and thread_id < 12:
            for i in range(0, 8):
                matrix[thread_id - 8][i + 8] = dictionar[thread_id][i]

        elif thread_id >= 12 and thread_id < 16:
            for i in range(0, 8):
                matrix[thread_id - 4][i + 8] = dictionar[thread_id][i]

        elif thread_id >= 16 and thread_id < 20:
            for i in range(0, 8):
                matrix[thread_id - 12][i] = dictionar[thread_id][i]

        elif thread_id >= 20 and thread_id < 24:
            for i in range(0, 8):
                matrix[thread_id - 8][i] = dictionar[thread_id][i]

        elif thread_id >= 24 and thread_id < 28:
            for i in range(0, 8):
                matrix[thread_id - 20][i + 8] = dictionar[thread_id][i]

        elif thread_id >= 28 and thread_id < 32:
            for i in range(0, 8):
                matrix[thread_id - 16][i + 8] = dictionar[thread_id][i]
                print(matrix)

    # store the matrix per row as the correct results from the operative point
    for j in range(0, 16):
        for i in range(0, 16):
            print(matrix[j][i])
        print("\n")

    print("******************")

    for j in range(0, 16):
        for i in range(0, 16):
            print(matrix[j][i])
            # store per columns as a flat memory of the results


# ----------------------------------------------------------------------------------------------------------

# preliminary evaluation posit 32
# *****************************************************************************************************

# 	value = Posit32(0.0)

# 	fault_model = "sa0"

# 	name = "text_file32_" + str(fault_model) + "_"

# 	for pointer in range (0, len(total_masks_32)):
##	for pointer in range (0, 1):

# 		print("eval: " + str(total_masks_32[pointer]) )
# 		j = -2.0
# 		file_output = open(name + str(pointer)  + ".txt", "w")

# 		for i in range(0, 4002, 1):
# 			P_32bits1 = Posit32(j)

# 			value = inject_fault_posit(P_32bits1 ,total_masks_32[pointer], fault_model, 1)

# 			file_output.write(str(j) + " " + str(P_32bits1) + " " + str(value) + "\n")
# 			j = j + 0.001

# 		file_output.close()

# *****************************************************************************************************


# preliminary evaluation posit 16
# *****************************************************************************************************

# 	value = Posit16(0.0)

# 	fault_model = "sa1"

# 	name = "text_file16_" + str(fault_model) + "_"

# 	for pointer in range (0, len(total_masks_16)):
##	for pointer in range (0, 1):

# 		print("eval: " + str(total_masks_16[pointer]) )
# 		j = -2.0
# 		file_output = open(name + str(pointer)  + ".txt", "w")

# 		for i in range(0, 4002, 1):
# 			P_16bits1 = Posit16(j)

# 			value = inject_fault_posit(P_16bits1 ,total_masks_16[pointer], fault_model, 1)

# 			file_output.write(str(j) + " " + str(P_16bits1) + " " + str(value) + "\n")
# 			j = j + 0.001

# 		file_output.close()

# *****************************************************************************************************

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


main()
