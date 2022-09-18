from readFile import *


def heat_map_pixels(size, amount, moves, numDict, pathBoards):
    """make a heat map of num pixels live in the data of games"""
    table = np.zeros(20, 20)
    for i in range(amount):
        bin_array = read_file_bin_array(path_file, size, amount)
        for j in range(SIZE):
            for k in range(SIZE):
                if bin_array[20*i+k]:
                    table[i, k] += 1
    return table


heat_map_pixels(SIZE, AMOUNT_BOARDS, AMOUNT_MOVES, NUM_DICT, PATH_BOARDS)
