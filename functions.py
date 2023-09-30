import itertools
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

SIZE = 10
AMOUNT_BOARDS = 100000
AMOUNT_MOVES = 100
NUM_DICT = 1
READ_FILE = 5
IGNORE_RANGE = 5


M = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
LEN = SIZE**2
PATH_BOARDS = 'C:\\GameOfLife\\boards\\'
BYTE = 8
PATH_IMAGES = 'C:\\GameOfLife\\images\\'
FILE_TO_READ = f"{SIZE}-{READ_FILE}-{AMOUNT_MOVES}boards.bnr"
PATH_TO_READ = str(READ_FILE % NUM_DICT) + "\\" + FILE_TO_READ


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
            if (
                col_idx == n
                and field[0][0]
                or col_idx != n
                and field[0][col_idx]
            ):
                neighs += 1
        elif col_idx == n:
            if field[row_idx][0]:
                neighs += 1
        elif field[row_idx][col_idx]:
            neighs += 1
    return neighs


def make_move(field, moves=1):
    """ Make a move forward according to Game of Life rules """
    n = len(field)
    cur_field = field[:]
    for _ in range(moves):
        new_field = [[0] * n for _ in range(n)]
        for i, j in itertools.product(range(n), range(n)):
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
    return sum(lst) / len(lst)


def min_no_zero(arr):
    return next((i for i in range(len(arr)) if arr[i] != 0), -1)


def max_no_zero(arr):
    return next((len(arr) - i for i in range(len(arr)) if arr[-i] != 0), -1)


def path(size, number, amount_moves, num_dict):
    name_file = f"{str(size)}-{str(number)}-{str(amount_moves)}boards.bnr"
    return str(number % num_dict) + "\\" + name_file


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
    list_boards = [
        str_boards[i * LEN : (i + 1) * LEN]
        for i in range(len(str_boards) // length)
    ]
    # print(list_boards)
    return list_boards, len(str_boards) // length


def conv_bin_array_to_str(bin_array):
    """the function convert bin array to string"""
    return ''.join(bin(elem)[2:].zfill(BYTE) for elem in bin_array)

def print_numbers(i):
    print(i, end=' ')
    if i % 50 == 0 and i != 0:
        print()