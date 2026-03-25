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
from sklearn.model_selection import train_test_split
import tensorflow as tf


SIZE = 10
AMOUNT_BOARDS = 10000
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


def find_loc(pixel, size):
    x = pixel // size
    y = pixel % size
    return (x,y)

def find_pixel(i,j,size):
    return size*i+j

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
    
    
def load_reverse_df(size, amount_boards, gen):
    name_df = f'{PATH_DF}\\{size}-{amount_boards}\\{size}size_{amount_boards}boards_{gen}gen_reverse'
    reverse_df = pd.read_pickle(f'{name_df}.pkl')
    new_columns = [f'Col_{i}' for i in range(gen*size*size)]
    reverse_df_sort = reverse_df.sort_values(by = new_columns).reset_index(drop=True)
    for i in reverse_df_sort.columns:
        reverse_df_sort[i] = reverse_df_sort[i].astype(int)
    return reverse_df_sort

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
    return X_train, X_test, y_train, y_test


def prepare_reverse_dataset(reverse_df, size, gen, target_pixel_index=0, test_size=0.1, val_size=0.1, random_state=365):
    """Prepare train/val/test split for reverse Game of Life data.

    Args:
        reverse_df (DataFrame): reverse dataset reading with gen*size*size columns
        size (int): board side length
        gen (int): number of generations in reverse chain
        target_pixel_index (int): index of target pixel in the final board (0-based)
        test_size (float): test split fraction
        val_size (float): validation fraction (of train_val)
        random_state (int): random state

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test (DataFrames/Series)
    """
    # All columns to integer
    cols = [f'Col_{i}' for i in range(gen * size * size)]
    reverse_df = reverse_df.copy()
    reverse_df[cols] = reverse_df[cols].astype(int)

    amount_features = len(reverse_df.columns) - size * size
    features = reverse_df.iloc[:, :amount_features]
    target_col = f'Col_{amount_features + target_pixel_index}'
    target = reverse_df[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_numpy_4d(X_train, X_val, X_test, y_train, y_val, y_test, size):
    """Convert pandas splits to numpy arrays reshaped for CNN input."""
    X_train_a = X_train.to_numpy().reshape((-1, size, size, 1))
    X_val_a = X_val.to_numpy().reshape((-1, size, size, 1))
    X_test_a = X_test.to_numpy().reshape((-1, size, size, 1))

    y_train_a = y_train.to_numpy().reshape((-1, 1))
    y_val_a = y_val.to_numpy().reshape((-1, 1))
    y_test_a = y_test.to_numpy().reshape((-1, 1))

    return X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a


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


import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_test_array, y_test_array):

    # predict test
    y_pred = model.predict(X_test_array)
    y_pred = y_pred.flatten()  # Ensure 1D array
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_test_array = y_test_array.flatten()  # Ensure 1D array

    # Confusion matrix
    cm = confusion_matrix(y_test_array, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # calc the parameters
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn)

    # print in table
    print("\n===== Evaluation Results =====")
    print("┌──────────────┬────────────┬────────────┐")
    print("│              │ Pred=Alive │ Pred=Dead  │")
    print("├──────────────┼────────────┼────────────┤")
    print(f"│ True=Alive   │ {tp:10d} │ {fn:10d} │")
    print(f"│ True=Dead    │ {fp:10d} │ {tn:10d} │")
    print("└──────────────┴────────────┴────────────┘")

    print("\n--- Performance Metrics ---")
    print(f"{'Accuracy':<12}: {acc:.3f}")
    print(f"{'Precision':<12}: {precision:.3f}")
    print(f"{'Recall':<12}: {recall:.3f}")
    print(f"{'F1-score':<12}: {f1:.3f}")


def build_and_train_nn(X_train_array, y_train_array, size, dense_units=(128, 64), epochs=10, batch_size=32, validation_split=0.2):
    """Build and train a small MLP/FC model for Game of Life reverse prediction."""
    import tensorflow as tf

    input_shape = (size, size, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units[0], activation='relu'),
        tf.keras.layers.Dense(dense_units[1], activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_array, y_train_array,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size)

    return model, history

def build_and_train_cnn(X_train_array, y_train_array, size, epochs=10, batch_size=32, validation_split=0.2):
    """Build and train a simple CNN for Game of Life reverse prediction."""
    import tensorflow as tf

    input_shape = (size, size, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_array, y_train_array,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size)

    return model, history

def build_and_train_rnn(X_train_array, y_train_array, size, gen, rnn_units=128, dense_units=64, epochs=20, batch_size=32, validation_split=0.2):
    """Build and train a simple RNN (LSTM) for Game of Life reverse prediction."""
    import tensorflow as tf

    input_dim = size * size
    timesteps = gen - 1

    # expected input shape from to_numpy_4d is (n, size, size, 1)
    X_rnn = X_train_array.reshape((-1, timesteps, input_dim)).astype('float32')
    y_rnn = y_train_array.astype('float32')

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(timesteps, input_dim)),
        tf.keras.layers.LSTM(rnn_units, activation='tanh'),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    history = model.fit(X_rnn, y_rnn,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size)

    return model, history

def build_RCNN_sidmoind(gen, x_train, y_train, size, batch_size, epochs):
    # --- PREPROCESSING ---
    gen_data = gen-1
    num_samples = x_train.shape[0] - gen_data

    X_train = np.zeros((num_samples, gen_data, size, size, 1), dtype='float32')
    Y_train = np.zeros((num_samples, 1), dtype='float32')  # רק תא אחד

    for i in range(num_samples):
        X_train[i] = x_train[i:i+gen_data].reshape(gen_data, size, size, 1)   # רצף של gen_data לוחות
        Y_train[i] = y_train[i]              # הפלט: תא אחד (0/1)

    print("X_train shape:", X_train.shape)  # (num_samples, gen_data, SIZE, SIZE, 1)
    print("y_train shape:", Y_train.shape)  # (num_samples, 1)

    # --- MODEL ---
    model = tf.keras.Sequential([
        tf.keras.layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3,3),
            activation='relu',
            padding='same',
            return_sequences=True,
            input_shape=(gen_data, size, size, 1)
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3,3),
            activation='relu',
            padding='same',
            return_sequences=False
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # ← פלט יחיד בינארי
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # אימון
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True
    )
    
    return model, history