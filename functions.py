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
    # downcast to int8 first to save memory (values are 0/1)
    for col in reverse_df.columns:
        reverse_df[col] = reverse_df[col].astype(np.int8)
    new_columns = [f'Col_{i}' for i in range(gen*size*size)]
    reverse_df.sort_values(by=new_columns, inplace=True, ignore_index=True)
    return reverse_df

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

    # Use gen-1 boards as features, last board as target
    amount_features = (gen - 1) * size * size
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


def to_numpy_4d(X_train, X_val, X_test, y_train, y_val, y_test, size, gen):
    """Convert pandas splits to numpy arrays reshaped for CNN input."""
    X_train_a = X_train.to_numpy().reshape((-1, size, size, gen-1))
    X_val_a = X_val.to_numpy().reshape((-1, size, size, gen-1))
    X_test_a = X_test.to_numpy().reshape((-1, size, size, gen-1))

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

def evaluate_model(model, X_test_array, y_test_array, size=None, gen=None, gen_data_override=None):
    """
    Evaluate model with automatic data reshaping based on model input shape.
    
    Auto-detects model type by inspecting model.input_shape:
    - 4D (*, size, size, 1) → NN/CNN (no reshaping needed)
    - 3D (*, timesteps, flat_dim) → RNN (reshape from (n,size,size,1) to (-1, timesteps, size*size))
    - 5D (*, gen_data, size, size, 1) → RCNN/ConvLSTM (build sliding window sequences)
    
    Args:
        model: Keras/TF model
        X_test_array: Test input features, shape (n, size, size, 1) typically
        y_test_array: Test labels
        size: Board dimension (e.g., 5). Required for 3D/5D reshaping.
        gen: Number of generations. Used to infer timesteps (gen-1). Required for 3D/5D reshaping.
        gen_data_override: Optional override for gen_data value if different from gen-1
    """
    # Flatten y_test if needed
    y_test_flat = y_test_array.flatten() if y_test_array.ndim > 1 else y_test_array
    
    # Get model input shape
    input_shape = model.input_shape
    ndim = len(input_shape)
    
    X_prepared = X_test_array.copy()
    
    # Auto-detect and reshape based on model input requirements
    if ndim == 4:
        # 4D input: (*, size, size, 1) - NN/CNN
        # No reshaping needed, use as-is
        pass
    
    elif ndim == 3:
        # 3D input: (*, timesteps, flat_dim) - RNN/LSTM
        if size is None or gen is None:
            raise ValueError("For RNN models, 'size' and 'gen' parameters are required")
        
        timesteps = gen - 1
        flat_dim = size * size
        
        if X_prepared.ndim == 4:  # Currently (n, size, size, 1)
            X_prepared = X_prepared.reshape((-1, timesteps, flat_dim)).astype('float32')
        elif X_prepared.ndim == 5:  # Currently (n, gen-1, size, size, 1)
            # Already in sequence form, just flatten the spatial dims
            n_samples = X_prepared.shape[0]
            X_prepared = X_prepared.reshape((n_samples, timesteps, flat_dim)).astype('float32')
    
    elif ndim == 5:
        # 5D input: (*, gen_data, size, size, 1) - RCNN/ConvLSTM
        if size is None or gen is None:
            raise ValueError("For RCNN models, 'size' and 'gen' parameters are required")
        
        gen_data = gen_data_override if gen_data_override is not None else (gen - 1)
        
        if X_prepared.ndim == 4:
            # Build sequences
            num_samples = X_prepared.shape[0] - gen_data
            X_new = np.zeros((num_samples, gen_data * gen_data, size, size, 1), dtype='float32')
            
            for i in range(num_samples):
                X_new[i] = X_test_array[i:i+gen_data].reshape(gen_data * gen_data, size, size, 1)
            
            X_prepared = X_new
            y_test_flat = y_test_flat[:num_samples]  # Trim y to match
        # else: already (n, gen_data*gen_data, size, size, 1), use as-is
    
    else:
        raise ValueError(f"Unsupported model input shape: {input_shape}")
    
    # predict test
    y_pred = model.predict(X_prepared)
    
    # Handle different output formats
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        # Multi-class or softmax output
        y_pred_binary = np.argmax(y_pred, axis=1).astype(int)
    else:
        # Sigmoid output
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Ensure y_test_flat matches y_pred_binary length
    y_test_binary = y_test_flat[:len(y_pred_binary)].astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_binary, y_pred_binary)
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


def _default_training_callbacks(patience=3, min_lr=1e-5):
    """Common callbacks for stable training and less overfitting."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=min_lr,
            verbose=1,
        ),
    ]


def _compute_binary_class_weight(y):
    """Return balanced class weights for binary labels, or None if unavailable."""
    y_flat = np.asarray(y).reshape(-1).astype(int)
    total = len(y_flat)
    if total == 0:
        return None

    count_0 = int(np.sum(y_flat == 0))
    count_1 = int(np.sum(y_flat == 1))
    if count_0 == 0 or count_1 == 0:
        return None

    return {
        0: total / (2.0 * count_0),
        1: total / (2.0 * count_1),
    }

def build_and_train_nn(X_train_array, y_train_array, size, dense_units=(128, 64), epochs=10, batch_size=32, validation_split=0.2,
                       use_callbacks=True, use_class_weight=True, fit_verbose=1):
    """Build and train a small MLP/FC model for Game of Life reverse prediction."""
    import tensorflow as tf

    input_shape = X_train_array.shape[1:]
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

    callbacks = _default_training_callbacks() if use_callbacks else None
    class_weight = _compute_binary_class_weight(y_train_array) if use_class_weight else None

    history = model.fit(X_train_array, y_train_array,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        verbose=fit_verbose)

    return model, history

def build_and_train_cnn(X_train_array, y_train_array, size, epochs=10, batch_size=32, validation_split=0.2,
                        use_callbacks=True, use_class_weight=True, fit_verbose=1):
    """Build and train a simple CNN for Game of Life reverse prediction."""
    import tensorflow as tf

    # Extract input shape from data (handles variable number of channels/generations)
    input_shape = X_train_array.shape[1:]
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

    callbacks = _default_training_callbacks() if use_callbacks else None
    class_weight = _compute_binary_class_weight(y_train_array) if use_class_weight else None

    history = model.fit(X_train_array, y_train_array,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        verbose=fit_verbose)

    return model, history

def build_and_train_rnn(X_train_array, y_train_array, size, gen, rnn_units=128, dense_units=64, epochs=20, batch_size=32,
                        validation_split=0.2, use_callbacks=True, use_class_weight=True, fit_verbose=1):
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

    callbacks = _default_training_callbacks() if use_callbacks else None
    class_weight = _compute_binary_class_weight(y_rnn) if use_class_weight else None

    history = model.fit(X_rnn, y_rnn,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        verbose=fit_verbose)

    return model, history


def build_and_train_rnn_v2(X_train_array, y_train_array, size, gen,
                           lstm_units_1=128, lstm_units_2=64,
                           dense_units_1=128, dense_units_2=64,
                           dropout_rate=0.3, recurrent_dropout=0.2,
                           learning_rate=0.001,
                           epochs=40, batch_size=32,
                           validation_split=0.2,
                           use_callbacks=True, use_class_weight=True, fit_verbose=1):
    """
    Improved RNN (Bidirectional stacked LSTM) for Game of Life reverse prediction.

    Improvements over build_and_train_rnn:
    1. Bidirectional LSTM — processes the board sequence forward AND backward,
       capturing temporal dependencies in both directions.
    2. Stacked 2-layer LSTM — deeper recurrent network extracts richer
       temporal features from the board evolution sequence.
    3. Dropout + Recurrent Dropout — regularization inside and between LSTM
       layers to reduce overfitting.
    4. BatchNormalization — stabilizes activations between layers for faster,
       more stable convergence.
    5. Larger Dense head (128 → 64) with Dropout — more decoding capacity
       while still regularized.
    6. Tunable learning rate — allows fine-tuning Adam's step size.
    7. More epochs (40 default) — combined with EarlyStopping, the model gets
       more chances to find a good minimum.
    """
    import tensorflow as tf

    input_dim = size * size
    timesteps = gen - 1

    X_rnn = X_train_array.reshape((-1, timesteps, input_dim)).astype('float32')
    y_rnn = y_train_array.astype('float32')

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(timesteps, input_dim)),

        # --- Layer 1: Bidirectional LSTM (return sequences for stacking) ---
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units_1, activation='tanh',
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        # --- Layer 2: Bidirectional LSTM (final recurrent output) ---
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units_2, activation='tanh',
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=False)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        # --- Dense head ---
        tf.keras.layers.Dense(dense_units_1, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(dense_units_2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = _default_training_callbacks(patience=5) if use_callbacks else None
    class_weight = _compute_binary_class_weight(y_rnn) if use_class_weight else None

    history = model.fit(X_rnn, y_rnn,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        verbose=fit_verbose)

    return model, history


def build_and_train_crnn_v3(X_train_array, y_train_array, size, gen,
                            conv_filters=(32, 64),
                            kernel_size=(3, 3),
                            lstm_units_1=128, lstm_units_2=64,
                            dense_units_1=128, dense_units_2=64,
                            dropout_rate=0.3, recurrent_dropout=0.2,
                            learning_rate=0.001, weight_decay=1e-4,
                            epochs=50, batch_size=512,
                            validation_split=0.2,
                            use_callbacks=True, use_class_weight=True, fit_verbose=1):
    """
    CRNN (Convolutional Recurrent Neural Network) for Game of Life reverse prediction.

    Architecture:
    1. TimeDistributed Conv2D blocks — extract local spatial features
       (Game of Life rules are local: 3×3 neighbourhood).
    2. Bidirectional stacked LSTM — capture sequential/temporal dependencies.
    3. Dense head with BatchNorm + Dropout — decode to binary prediction.

    Key improvements over v2:
    - Conv2D layers BEFORE the recurrent layers add spatial awareness.
    - AdamW optimizer with weight decay for better generalisation.
    - ReduceLROnPlateau callback included by default.
    - Sigmoid output for binary cell classification.
    """
    import tensorflow as tf

    input_dim = size * size
    timesteps = gen - 1

    # Reshape: (n, size, size, gen-1) → (n, timesteps, size, size, 1)
    X_crnn = X_train_array.reshape((-1, timesteps, size, size, 1)).astype('float32')
    y_crnn = y_train_array.astype('float32')

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(timesteps, size, size, 1)),

        # --- Spatial feature extraction (applied per timestep) ---
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(conv_filters[0], kernel_size, activation='relu', padding='same')
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(conv_filters[1], kernel_size, activation='relu', padding='same')
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(dropout_rate)),

        # --- Recurrent layers (Bidirectional stacked LSTM) ---
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units_1, activation='tanh',
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units_2, activation='tanh',
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=False)
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        # --- Dense head ---
        tf.keras.layers.Dense(dense_units_1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(dense_units_2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks: EarlyStopping + ReduceLROnPlateau
    callbacks = None
    if use_callbacks:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=7,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=3, min_lr=1e-6, verbose=1
            ),
        ]

    class_weight = _compute_binary_class_weight(y_crnn) if use_class_weight else None

    history = model.fit(X_crnn, y_crnn,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        class_weight=class_weight,
                        verbose=fit_verbose)

    return model, history


def build_and_train_rcnn(gen, x_train, y_train, size, batch_size, epochs, active = "sigmoid",
                         use_callbacks=True, use_class_weight=True, fit_verbose=1):
    # --- PREPROCESSING ---
    gen_data = gen-1
    num_samples = x_train.shape[0] - gen_data

    X_train = np.zeros((num_samples, gen_data*gen_data, size, size, 1), dtype='float32')
    if active == "softmax":
        Y_train = np.zeros((num_samples, 2), dtype='float32')  # one-hot for softmax
    else:
        Y_train = np.zeros((num_samples, 1), dtype='float32')  # binary for sigmoid

    for i in range(num_samples):
        X_train[i] = x_train[i:i+gen_data].reshape(gen_data*gen_data, size, size, 1)   # רצף של gen_data לוחות
        if active == "softmax":
            Y_train[i] = [1, 0] if y_train[i] == 0 else [0, 1]  # one-hot
        else:
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
            input_shape=(gen_data * gen_data, size, size, 1)
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
        tf.keras.layers.Dense(2 if active == "softmax" else 1, activation=active)  
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy' if active == "softmax" else 'binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # אימון
    callbacks = _default_training_callbacks() if use_callbacks else None
    class_weight = None
    if active != "softmax" and use_class_weight:
        class_weight = _compute_binary_class_weight(Y_train)

    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        shuffle=True,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=fit_verbose,
    )
    
    return model, history

'''def build_RCNN_sigmoid(gen, x_train, y_train, size, batch_size=32, epochs=3, active = "sigmoid"):
    # sourcery skip: inline-immediately-returned-variable
    # --- פרמטרים ---
    gen_data = gen - 1    # מספר הלוחות הרציפים בקלט

    # --- PREPROCESSING ---
    # X_train_array.shape = (num_samples + gen_data, SIZE, SIZE, 1)
    # y_train_array.shape = (num_samples, 1)  ← תא אחד בלבד (0 או 1)

    num_samples = x_train.shape[0] - gen_data

    X_train = np.zeros((num_samples, gen_data*gen_data, size, size, 1), dtype='float32')
    Y_train = np.zeros((num_samples, 1), dtype='float32')  # רק תא אחד

    for i in range(num_samples):
        X_train[i] = x_train[i:i+gen_data].reshape(gen_data*gen_data, size, size, 1)   # רצף של gen_data לוחות
        Y_train[i] = y_train[i]              # הפלט: תא אחד (0/1)

    print("X_train shape:", X_train.shape)  # (num_samples, gen_data*gen_data, size, size, 1)
    print("y_train shape:", y_train.shape)  # (num_samples, 1)

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
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # ***** build RNN/RCNN train/test arrays *****
    num_samples_train = X_train.shape[0] - gen_data

    X_train_rnn = np.zeros((num_samples_train, gen_data, size, size, 1), dtype='float32')
    y_train_rnn = np.zeros((num_samples_train,), dtype='float32')
    for i in range(num_samples_train):
        X_train_rnn[i] = X_train[i:i + gen_data].reshape(gen_data, size, size, 1)
        y_train_rnn[i] = y_train[i].astype('float32')

    # אימון
    history = model.fit(
        X_train_rnn,
        y_train_rnn,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True
    )
    
    return model, history'''