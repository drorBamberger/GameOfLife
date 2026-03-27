import numpy as np
import pandas as pd
from functions import *
from datetime import datetime

def read_file_to_df(pathFile, size):
    # Define the file path
    #path_file = 'C:\\GameOfLife\\boards\\' + path(SIZE, READ_FILE, AMOUNT_MOVES, NUM_DICT)
    path_file = PATH_BOARDS + pathFile

    # Read binary data into a numpy array of 8-bit integers
    binary_data = np.fromfile(path_file, dtype=np.uint8)

    # Convert each integer to its binary representation
    binary_data_binary_repr = np.array([list(np.binary_repr(x, width=8)) for x in binary_data], dtype=np.uint8).flatten()

    # Calculate the total number of elements in each game board
    board_elements = size**2

    # Calculate the number of boards in the file
    num_boards = len(binary_data_binary_repr) // board_elements

    # Reshape the binary data to create a 3D array with dimensions (num_boards, board_size[0], board_size[1])
    boards_data = binary_data_binary_repr[:num_boards * board_elements].reshape((num_boards, size, size))

    # Create a list of column names based on the board size
    column_names = [f'({i},{j})' for i in range(size) for j in range(size)]

    return pd.DataFrame(boards_data.reshape((num_boards, board_elements)), columns=column_names)

def split_board_to_series_df(size, amount_boards, amount_moves, num_dict, amount_board_in_series, ignore_range, reverse = False):
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    new_columns = [f'Col_{i}' for i in range(amount_board_in_series*size*size)]
    res_df = pd.DataFrame(columns=new_columns)

    for i in range(amount_boards):
        print_big_numbers(i)
        # path to read
        path_file = path(size, i, amount_moves, num_dict, amount_boards)
        # read the file
        df = read_file_to_df(path_file, size)
        #after we delete the repeat boards, we split the board to series
        if(len(df)>ignore_range):
            df = df[ignore_range:].drop_duplicates().reset_index(drop=True)
            new_row = df
            for _ in range(amount_board_in_series-1):
                new_col = df.iloc[1:].reset_index(drop=True)
                df = df.iloc[1:].reset_index(drop=True)
                if reverse==False:
                    new_row = pd.concat([new_row.iloc[:-1],new_col],axis=1)
                else:
                    new_row = pd.concat([new_col,new_row.iloc[:-1]],axis=1)
            new_row.columns = new_columns
            res_df = pd.concat([res_df,new_row])
    return res_df


def _build_time_series_from_array(arr, gen, reverse=False):
    """Build sliding windows (gen frames) from board array matrix."""
    n, d = arr.shape
    if n < gen:
        return np.empty((0, gen * d), dtype=arr.dtype)

    n_seq = n - gen + 1
    # shape (n_seq, gen, d)
    windows = np.stack([arr[i:i + n_seq] for i in range(gen)], axis=1)
    if reverse:
        windows = windows[:, ::-1, :]
    return windows.reshape(n_seq, gen * d)


def split_board_to_series_df_del(size, amount_boards, amount_moves, num_dict, amount_board_in_series, ignore_range, reverse=False):
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    new_columns = [f'Col_{i}' for i in range(amount_board_in_series * size * size)]

    res_chunks = []
    for i in range(amount_boards):
        print_big_numbers(i)
        path_file = path(size, i, amount_moves, num_dict, amount_boards)
        df = read_file_to_df(path_file, size)

        if len(df) > ignore_range:
            df = df[ignore_range:].drop_duplicates().reset_index(drop=True)
            arr = df.to_numpy(dtype=np.uint8)
            series_arr = _build_time_series_from_array(arr, amount_board_in_series, reverse=reverse)

            if series_arr.size == 0:
                continue

            res_chunks.append(pd.DataFrame(series_arr, columns=new_columns))

        del df

    if not res_chunks:
        return pd.DataFrame(columns=new_columns)

    return pd.concat(res_chunks, ignore_index=True)


def dec_tree_df(X_train, y_train, X_test, y_test, md = None ,rs = 42):
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

    return dt, dt.tree_.node_count, dt.tree_.max_depth, train_test_full_error
