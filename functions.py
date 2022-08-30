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

M = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
SIZE = 10
AMOUNT = 100
MOVES = 5

num_dict = 1
path_boards = 'C:\\GameOfLife\\boards\\'


# the calcUneighs, make_move and generate_population functions from:
# https://medium.com/@ptyshevs/rgol-ga-1cafc67db6c7


def calc_neighs(field, i, j):
    """ Calculate number of neighbors alive (assuming square field) """
    neighs = 0
    n = len(field)
    for m in M:
        row_idx = m[0] + i
        col_idx = m[1] + j
        if 0 <= row_idx < n and 0 <= col_idx < n:
            if field[row_idx][col_idx]:
                neighs += 1
    return neighs


def make_move(field, moves=1):
    """ Make a move forward according to Game of Life rules """
    n = len(field)
    cur_field = field
    for _ in range(moves):
        new_field = [[0]*n]*n
        for i in range(n):
            for j in range(n):
                neighs = calc_neighs(cur_field, i, j)
                if cur_field[i][j] and neighs == 2:
                    new_field[i][j] = 1
                if neighs == 3:
                    new_field[i][j] = 1
        cur_field = new_field
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

