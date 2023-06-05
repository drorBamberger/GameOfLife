from functions import *
from matplotlib import pyplot as plt


def add_board_to_heat_map(heat_map, size, field):
    """add a board to heat map"""
    for i in range(size):
        for j in range(size):
            if field[size * i + j] == '1':
                heat_map[i, j] += 1
    return heat_map


def print_heat_map(heat_map):
    print()
    print(heat_map)
    # plt.imshow(heat_map,cmap='hot', interpolation='nearest')
    # plt.show()

    # plt.figure(figsize=(15, 15))
    plt.title('Pixel frequency')
    sns.heatmap(heat_map, annot=True, fmt="d")
    plt.show()


# def heat_map_all():
#     heat_map = heat_map_pixels(SIZE, AMOUNT_BOARDS, AMOUNT_MOVES, NUM_DICT)
#     plt.imshow(heat_map, cmap='hot', interpolation='nearest')
#     plt.show()


def heat_map_pixels(size, amount_boards, amount_moves, num_dict):
    """make a heat map of num pixels live in the data of games"""
    length = size * size
    heat_map = np.zeros((size, size), dtype=int)
    list_amount_boards = dict()
    for i in range(amount_boards):
        print(i, end=' ')
        if i % 100 == 0:
            print()
        # path to read
        path_file = path(size, i, amount_moves, num_dict)
        # read the file
        boards, amount_boards = read_file_to_list(path_file, length)
        # add any board in the file to the heat map
        for board in boards:
            heat_map = add_board_to_heat_map(heat_map, size, board)
        list_amount_boards[i] = amount_boards
    return heat_map, list_amount_boards


def main():
    heat_map, dict_amount_boards = heat_map_pixels(SIZE, AMOUNT_BOARDS, AMOUNT_MOVES, NUM_DICT)
    print("heat map of pixels")
    print_heat_map(heat_map)

    # plt.plot(*zip(*sorted(dict_amount_boards.items())))
    # plt.show()

    # print(dict(sorted(dict_amount_boards.items(), key=lambda item: item[1])))

    dict_by_amount = {}
    for i in range(AMOUNT_MOVES + 2):
        dict_by_amount[i] = 0

    for value in dict_amount_boards.values():
        dict_by_amount[value] += 1

    print("amount of boards of each number")
    print(dict_by_amount)
    # plt.plot(*zip(*sorted(dict_by_amount.items())))
    # plt.show()

    keys = list(dict_by_amount.keys())
    values = list(dict_by_amount.values())
    plt.bar(keys, values)
    plt.show()

    loops = 0
    no_loops = 0
    for i in range(len(dict_amount_boards)):
        if dict_amount_boards[i] == AMOUNT_MOVES + 1:
            path_file = path(SIZE, i, AMOUNT_MOVES, NUM_DICT)
            boards, amount_boards = read_file_to_list(path_file, LEN)
            if boards[100] in boards[:100]:
                loops += 1
            else:
                no_loops += 1
    print("number of boards with loops:", loops)
    print("number of boards without loops:", no_loops)


if __name__ == "__main__":
    main()
