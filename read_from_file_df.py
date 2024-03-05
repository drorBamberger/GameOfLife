import numpy as np
import pandas as pd
from functions import *

def read_file_to_df(pathFile, size):
    # Define the file path
    #path_file = 'C:\\GameOfLife\\boards\\' + path(SIZE, READ_FILE, AMOUNT_MOVES, NUM_DICT)
    path_file = 'C:\\GameOfLife\\boards\\' + pathFile

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

def split_board_to_series_df(size, amount_boards, amount_moves, num_dict, amount_board_in_series, ignore_range):
    new_columns = [f'Col_{i}' for i in range(1, amount_board_in_series * size*size + 1)]
    res_df = pd.DataFrame(columns=new_columns)

    for i in range(amount_boards):
        print_numbers(i)
        # path to read
        path_file = path(size, i, amount_moves, num_dict)
        # read the file
        df = read_file_to_df(path_file, size)
        #after we delete the repeat boards, we split the board to series
        if(len(df)>ignore_range):
            #df = df[ignore_range:].drop_duplicates()
            #for i in range(len(df) - amount_board_in_series + 1):
            #    new_row = df.iloc[i:i + amount_board_in_series].values.flatten()
            #    new_df.loc[len(new_df)] = new_row     
            df = df[ignore_range:].drop_duplicates().reset_index(drop=True)
            df1 = df.iloc[:-1].reset_index(drop=True)
            df2 = df[1:].reset_index(drop=True)
            new_df = pd.concat([df1, df2], axis=1, ignore_index=True)
            new_df.columns = new_columns
            
            res_df = pd.concat([res_df,new_df])
    return res_df


def dec_tree_df(X_train,y_train, X_test, y_test, md = None ,rs = 42):
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