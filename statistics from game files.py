from readFile import *


def add_board_to_heat_map(heat_map, size, field):
    """add a board to heat map"""
    for i in range(size):
        for j in range(size):
            if field[size * i + j] == '1':
                heat_map[i, j] += 1
    return heat_map


def print_heat_map(heat_map):
    plt.imshow(heat_map, cmap='hot', interpolation='nearest')
    plt.show()
    # counts
    counts = df.apply(pd.value_counts).fillna(0)

    # plot
    sns.heatmap(counts, cmap="GnBu", annot=True)


def heat_map_all():
    heat_map = heat_map_pixels(SIZE, AMOUNT_BOARDS, AMOUNT_GENERATIONS, NUM_DICT, PATH_BOARDS)
    plt.imshow(heat_map, cmap='hot', interpolation='nearest')
    plt.show()


def heat_map_pixels(size, amount_boards, amount_generations, numDict, pathBoards):
    """make a heat map of num pixels live in the data of games"""
    len = size * size
    heat_map = np.zeros((size, size))
    for i in range(amount_boards):
        print(i, end=' ')
        if i % 100 == 0:
            print()
        # path to read
        path_file = path(size, i, amount_generations, numDict)
        # read the file
        boards = read_file_string(path_file, len, amount_generations)
        # add any board in the file to the heat map
        for j in range(amount_generations):
            heat_map = add_board_to_heat_map(heat_map, size, now_board(boards, len, j))
    return heat_map


def now_board(boards, len, j):
    return boards[j * len:(j + 1) * len]


def path(size, number, amount_generations, numDict):
    name_file = str(size) + "-" + str(number) + "-" + str(amount_generations) + "boards" + ".bnr"  # first board
    path_file = str(number % numDict) + "\\" + name_file
    return path_file


def main():
    heat_map = heat_map_pixels(SIZE, AMOUNT_BOARDS, AMOUNT_GENERATIONS, NUM_DICT, PATH_BOARDS)
    print()
    print(heat_map)
    print_heat_map(heat_map)


if __name__ == "__main__":
    main()
