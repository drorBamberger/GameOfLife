from functions import *


def read_file(path, size):
    length = size
    if length % 8 != 0:
        length = length + 8 - length % 8

    # read file
    file_path = path_boards + path
    with open(file_path, 'rb') as f:
        binArray = f.read(length // 8)
    return binArray


def conv_bin_array_to_str(binArray, size):
    # conv bin array to str
    res = ''
    for elem in binArray:
        res += bin(elem)[2:].zfill(8)

    # remove zeros
    if size % 8:
        res = res[:size]
    return res


def print_board(my_board, size):
    table = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            color = int(my_board[size * i + j])*-1 + 1
            table[i, j] = [color, color, color]
    plt.imshow(table, interpolation='nearest')
    plt.show()


bin_array = read_file('0\\5-39FB.bnr', SIZE * SIZE)
board = conv_bin_array_to_str(bin_array, SIZE * SIZE)
print(board)
print_board(board, SIZE)
