from functions import *

START = time.time()
num_dict = 1
path_boards = 'C:\\GameOfLife\\boards'


def rand_str(num, count):
    np.random.seed(num)
    res = ''
    for i in range(count):
        temp = str(random.randint(0, 1))
        res += temp
    print(res)
    return res


def conv_str_to_bin_array(str, size):
    # fill str with zeros
    length = size
    if length % 8:
        str += '0' * (8 - length % 8)
        length = length + 8 - length % 8

    # conv to bin array
    bin_array = array("B")
    for i in range(length // 8):
        bin_array.append(int(str[i * 8:(i + 1) * 8], 2))

    return bin_array


def create_board(num, size):
    boardStr = rand_str(num, size * size)
    boardBin = conv_str_to_bin_array(boardStr, size * size)
    return boardBin


# make directions
for i in range(num_dict):
    if os.path.isdir(path_boards + " " + str(i)):
        shutil.rmtree(path_boards + " " + str(i))
    os.mkdir(path_boards + " " + str(i))

# fill the files with boards
for i in range(AMOUNT):
    name = str(SIZE) + "-" + str(i) + "FB" + ".bnr"  # first board
    board = create_board(1, SIZE)
    with open(path_boards + " " + str(i % num_dict) + "\\" + name, 'wb') as f:
        f.write(bytes(board))
