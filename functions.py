import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from array import array
import shutil
import sys
from PIL import Image
from matplotlib import cm
from PIL.Image import Resampling
import seaborn as sns
import pandas as pd
import igraph

SIZE = 10
AMOUNT_BOARDS = 100000
AMOUNT_MOVES = 100
NUM_DICT = 10
READFILE = 9996

M = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
LEN = SIZE * SIZE
PATH_BOARDS = 'C:\\GameOfLife\\boards\\'
BYTE = 8
PATH_IMAGES = 'C:\\GameOfLife\\images\\'
FILE_TO_READ = str(SIZE) + "-" + str(READFILE) + "-" + str(AMOUNT_MOVES) + "boards" + ".bnr"  # first board
PATH_TO_READ = str(READFILE % NUM_DICT) + "\\" + FILE_TO_READ


# the calcUneighs, make_move and generate_population functions from:
# https://medium.com/@ptyshevs/rgol-ga-1cafc67db6c7


def calc_neighs(field, i, j):
    """ Calculate number of neighbors alive (assuming square field) """
    neighs = 0
    n = len(field)
    for m in M:
        row_idx = m[0] + i
        col_idx = m[1] + j
        if row_idx == n:
            if col_idx == n:
                if field[0][0]:
                    neighs += 1
            else:
                if field[0][col_idx]:
                    neighs += 1
        elif col_idx == n:
            if field[row_idx][0]:
                neighs += 1
        else:
            if field[row_idx][col_idx]:
                neighs += 1
    return neighs


def make_move(field, moves=1):
    """ Make a move forward according to Game of Life rules """
    n = len(field)
    cur_field = field[:]
    for l in range(moves):
        new_field = []
        for i in range(n):
            new_field.append([0] * n)
        for i in range(n):
            for j in range(n):
                neighs = calc_neighs(cur_field, i, j)
                if cur_field[i][j] and neighs == 2:
                    new_field[i][j] = 1
                elif neighs == 3:
                    new_field[i][j] = 1
        cur_field = new_field[:]
    return cur_field


def deduction_edges(lst):
    for i in lst:
        if sum(sum(i)) < SIZE * SIZE * 0.05:
            return []
        elif sum(sum(i)) > SIZE * SIZE * 0.95:
            return []
        else:
            return i


def Average(lst):
    avg = sum(lst) / len(lst)
    return avg


def min_no_zero(arr):
    for i in range(len(arr)):
        if arr[i] != 0:
            return i
    return -1


def max_no_zero(arr):
    for i in range(len(arr)):
        if arr[-i] != 0:
            return len(arr) - i
    return -1


def path(size, number, amount_moves, num_dict):
    name_file = str(size) + "-" + str(number) + "-" + str(amount_moves) + "boards" + ".bnr"  # first board
    path_file = str(number % num_dict) + "\\" + name_file
    return path_file


def read_file_bin_array(path):
    """the function read the bin file, and insert the board to bin array"""
    # read file
    file_path = PATH_BOARDS + path
    with open(file_path, 'rb') as f:
        bin_array = f.read()
    return bin_array


def read_file_to_list(path, length):
    """the function read the bin file, and insert the board to str array"""

    bin_array = read_file_bin_array(path)
    # print(bin_array)
    str_boards = conv_bin_array_to_str(bin_array)
    # print(str_boards)
    list_boards = list()
    # print(len(str_boards) // length)
    for i in range(len(str_boards) // length):
        # print(str_boards[i * LEN:(i + 1) * LEN])
        list_boards.append(str_boards[i * LEN:(i + 1) * LEN])
    # print(list_boards)
    return list_boards, len(str_boards) // length


def conv_bin_array_to_str(bin_array):
    """the function convert bin array to string"""
    # conv bin array to str
    res = ''
    for elem in bin_array:
        res += bin(elem)[2:].zfill(BYTE)
    return res