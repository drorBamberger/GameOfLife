from functions import *


def rand_board(size, random_state=-1):
    """
    Generating initial population of individual solutions
    :return: initial population as a list of 20x20 arrays
    """
    if random_state != -1:
        np.random.seed(random_state)
    return np.random.binomial(1, 0.5, (size, size)).tolist()


def flatten(lst):
    """flatten list to one list"""
    return [x for xs in lst for x in xs]


def convert_list_to_str(numList):  # [1,2,3]
    """convert list type to string"""
    s = map(str, numList)  # ['1','2','3']
    s = ''.join(s)  # '123'
    return s


def conv_lst_to_bin_array(lst):
    """convert string to bin array"""
    flat = flatten(lst)
    length = len(flat)

    # fill list with zeros
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
    """create boards and the next moves"""
    first_board = rand_board(size, num)
    boards = first_board
    now_board = first_board[:]
    next_board = []
    prev_board = []
    amount_boards = 1
    for _ in range(moves):
        next_board = make_move(now_board)
        if next_board not in [now_board, prev_board]:
            boards += next_board
            prev_board = now_board
            now_board = next_board
            amount_boards += 1
        else:
            break
    boards_bin = array("B")
    boards_bin += conv_lst_to_bin_array(boards)
    return boards_bin, amount_boards


def build_files(path, amount_boards, amount_moves, size, num_dict):
    # Delete the previous files
    if os.path.isdir(path[:-1]):
        shutil.rmtree(path[:-1])
    os.mkdir(path[:-1])

    # make directions for boards
    for i in range(num_dict):
        os.mkdir(path + str(i))

    # fill the files with boards
    for i in range(amount_boards):
        print_big_numbers(i)
        name = f"{str(size)}-{str(i)}-{str(amount_moves)}boards.bnr"
        board, a = create_board(i, size, amount_moves)
        with open(path + str(i % num_dict) + "\\" + name, 'wb') as f:
            # f.write(bytes(a))
            f.write(bytes(board))

def main():
    print(PATH_BOARDS_BY_SIZE, AMOUNT_BOARDS, AMOUNT_MOVES, SIZE, NUM_DICT)
    build_files(PATH_BOARDS_BY_SIZE, AMOUNT_BOARDS, AMOUNT_MOVES, SIZE, NUM_DICT)
    


if __name__ == "__main__":
    main()