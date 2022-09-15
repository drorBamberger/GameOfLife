from functions import *


def rand_board(size, random_state=-1):
    """
    Generating initial population of individual solutions
    :return: initial population as a list of 20x20 arrays
    """
    # print(size)
    if random_state != -1:
        np.random.seed(random_state)
    initial_states = np.random.binomial(1, 0.5, (size, size)).tolist()
    return initial_states


def flatten(lst):
    return [x for xs in lst for x in xs]


def convert_list_to_str(numList):  # [1,2,3]
    s = map(str, numList)  # ['1','2','3']
    s = ''.join(s)  # '123'
    return s


def conv_lst_to_bin_array(lst, size, count):
    flat = flatten(lst)
    # print(flat)

    # fill list with zeros
    length = size*size*count
    if length % 8 != 0:
        flat += [0] * (8 - length % 8)
        length = length + 8 - length % 8

    # conv to bin array
    bin_array = array("B")
    array_str = convert_list_to_str(flat)
    # print(array_str)
    for j in range(length // 8):
        bin_array.append(int(array_str[j * 8:(j + 1) * 8], 2))
    return bin_array


def create_board(num, size, moves=5):
    first_board = rand_board(size, num)
    # print(first_board)
    boards = first_board
    now_board = first_board[:]
    for i in range(moves):
        # print(now_board)
        now_board = make_move(now_board)
        # print(now_board)
        boards += now_board
        # print(boards)

    boardsBin = array("B")
    # print(boards)
    boardsBin += conv_lst_to_bin_array(boards, size, moves+1)
    # print(boardsBin)
    return boardsBin


if os.path.isdir(path_boards[:-1]):
    shutil.rmtree(path_boards[:-1])
os.mkdir(path_boards[:-1])

# make directions for boards
for i in range(num_dict):
    os.mkdir(path_boards + str(i))

# fill the files with boards
for i in range(AMOUNT):
    name = str(SIZE) + "-" + str(i) + "-6boards" + ".bnr"  # first board
    board = create_board(i, SIZE, MOVES)
    with open(path_boards + str(i % num_dict) + "\\" + name, 'wb') as f:
        # print(board)
        f.write(bytes(board))
