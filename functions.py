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

SIZE = 5
AMOUNT_BOARDS = 1000
AMOUNT_MOVES = 5
NUM_DICT = 10
READFILE = 1

M = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
LEN = SIZE * SIZE
AMOUNT_GENERATIONS = AMOUNT_MOVES + 1
PATH_BOARDS = 'C:\\GameOfLife\\boards\\'
BYTE = 8
PATH_IMAGES = 'C:\\GameOfLife\\images\\'
FILE_TO_READ = str(SIZE) + "-" + str(READFILE) + "-" + str(AMOUNT_GENERATIONS) + "boards" + ".bnr"  # first board
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
