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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree



SIZE = 10
AMOUNT_BOARDS = 1000
AMOUNT_MOVES = 100
NUM_DICT = 1
READ_FILE = 8
IGNORE_RANGE = 5

P = 5

LEN = SIZE**2
PATH_BOARDS = 'C:\\GameOfLifeFiles\\boards\\'
PATH_BOARDS_BY_SIZE = f"{PATH_BOARDS}\\boards_{SIZE}_{AMOUNT_BOARDS}_{AMOUNT_MOVES}_{NUM_DICT}\\"
PATH_IMAGES = 'C:\\GameOfLifeFiles\\images\\'
PATH_MODELS = 'C:\\GameOfLifeFiles\\models\\'
PATH_DATA_TEST = 'C:\\GameOfLifeFiles\\dataTest\\'
PATH_DF = 'C:\\GameOfLifeFiles\\df\\'
PATH_DF_BY_SIZE = f"{PATH_DF}\\{SIZE}-{AMOUNT_BOARDS}\\"

BYTE = 8
PATH_TO_READ = f"boards_{SIZE}_{AMOUNT_BOARDS}_{AMOUNT_MOVES}_{NUM_DICT}\\{str(READ_FILE % NUM_DICT)}\\{SIZE}-{READ_FILE}-{AMOUNT_MOVES}boards.bnr"


# the calcUneighs, make_move and generate_population functions from:
# https://medium.com/@ptyshevs/rgol-ga-1cafc67db6c7


def generate_twice_tuples(n):
    result = []
    for i in range(-n, n+1):
        result.extend((i, j) for j in range(-n, n+1))
    result.remove((0,0))
    return result


def calc_neighs(field, i, j, radii = 1):
    """ Calculate number of neighbors alive (assuming square field) """
    neighs = 0
    n = len(field)
    M = generate_twice_tuples(radii)
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
        elif field[row_idx][col_idx] == 1:
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


def path(size, number, amount_moves, num_dict, amount_boards):
    return f"boards_{size}_{amount_boards}_{amount_moves}_{num_dict}\\{str(number % num_dict)}\\{str(size)}-{str(number)}-{str(amount_moves)}boards.bnr"


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
    list_boards = [str_boards[i * LEN : (i + 1) * LEN] for i in range(len(str_boards) // length)]
    # print(list_boards)
    return list_boards, len(str_boards) // length


def conv_bin_array_to_str(bin_array):
    """the function convert bin array to string"""
    return ''.join(bin(elem)[2:].zfill(BYTE) for elem in bin_array)

def print_numbers(i):
    print(i, end=' ')
    if i % 50 == 0 and i != 0:
        print()
        
        
def print_big_numbers(i):
    if i % 50 == 0:
        print(i, end=' ')
    if i % 10000 == 0:
        print()
        
        
def measure_error(y_true, y_pred, label):
    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred),
                    'recall': recall_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred)},
                    name=label)
    
def prepare_data(df, percent_to_test):
    """_summary_

    Args:
        df (list): data for split
        percent_to_test (float): percent to test for split data

    Returns:
        X_train : list
        X_test  : list
        y_train  : list
        y_test  : list
    """
    X = []

    for i, line in enumerate(df):
        print_big_numbers(i)
        line_result = []
        for string in line[:-1]:
            line_result.extend([int(char) for char in string])
        X.append(line_result)

    y = [int(line[-1][0]) for line in df]

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(X)):
        print_big_numbers(i)
        if i%(1/percent_to_test) != 0:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    return X_train ,X_test, y_train, y_test

def dec_tree(X_train,y_train, X_test, y_test, md ,rs):
    """_summary_

    Args:
        X_train (list): 
        y_train (list): 
        X_test (list): 
        y_test (list): 
        md (int): max depth
        rs (int): random state
    """
    dt = tree.DecisionTreeClassifier(max_depth = md, random_state=rs)
    dt = dt.fit(X_train, y_train)

    # The error on the training and test data sets
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                                    measure_error(y_test, y_test_pred, 'test')],
                                    axis=1)

    print(dt.tree_.node_count, dt.tree_.max_depth)
    return dt, train_test_full_error