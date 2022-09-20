from functions import *


def rand_board(size, random_state=-1):
    """
    Generating initial population of individual solutions
    :return: initial population as a list of 20x20 arrays
    """
    if random_state != -1:
        np.random.seed(random_state)
    initial_states = np.random.binomial(1, 0.5, (size, size)).tolist()
    return initial_states


def flatten(lst):
    """flatten list to one list"""
    return [x for xs in lst for x in xs]


def convert_list_to_str(numList):  # [1,2,3]
    """convert list type to string"""
    s = map(str, numList)  # ['1','2','3']
    s = ''.join(s)  # '123'
    return s


def conv_lst_to_bin_array(lst, size, count):
    """convert string to bin array"""
    flat = flatten(lst)

    # fill list with zeros
    length = size * size * count
    if length % BYTE != 0:
        flat += [0] * (BYTE - length % BYTE)
        length = length + BYTE - length % BYTE

    # conv to bin array
    bin_array = array("B")
    array_str = convert_list_to_str(flat)
    for j in range(length // BYTE):
        bin_array.append(int(array_str[j * BYTE:(j + 1) * BYTE], 2))
    return bin_array


def create_board(num, size, moves=5):
    """crate boards and the next moves"""
    first_board = rand_board(size, num)
    boards = first_board
    now_board = first_board[:]
    for i in range(moves):
        now_board = make_move(now_board)
        boards += now_board

    boardsBin = array("B")
    boardsBin += conv_lst_to_bin_array(boards, size, moves + 1)
    return boardsBin


def main():
    # Delete the previous images
    if os.path.isdir(PATH_BOARDS[:-1]):
        shutil.rmtree(PATH_BOARDS[:-1])
    os.mkdir(PATH_BOARDS[:-1])

    # make directions for boards
    for i in range(NUM_DICT):
        os.mkdir(PATH_BOARDS + str(i))

    # fill the files with boards
    for i in range(AMOUNT_BOARDS):
        print(i, end=' ')
        if i % 100 == 0:
            print()
        name = str(SIZE) + "-" + str(i) + "-" + str(AMOUNT_GENERATIONS) + "boards" + ".bnr"  # first board
        board = create_board(i, SIZE, AMOUNT_MOVES)
        with open(PATH_BOARDS + str(i % NUM_DICT) + "\\" + name, 'wb') as f:
            f.write(bytes(board))


if __name__ == "__main__":
    main()
