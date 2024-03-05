from functions import *


def split_boards(boards, amount, ignore_range, amount_board_in_series):
    if boards_game[amount_boards_of_game - 1]  in boards_game[:amount_boards_of_game-1]: # there is loop
            boards_game, amount_boards_of_game  = delete_repeat(boards_game, amount_boards_of_game)
    boards = boards[ignore_range:]
    return [(boards[i:i + amount_board_in_series]) for i in range(len(boards) - amount_board_in_series + 1)]


def delete_repeat(boards, amount):
    # delete the repeat boards
    # print("there is loop")
    res = []
    for i in range(amount):
        if boards[i] in boards[i+1:]:
            amount -= 1
        else:
            res.append(boards[i])
    return res, amount


def split_board_to_series(size, amount_boards, amount_moves, num_dict, amount_board_in_series):
    series = []
    for i in range(amount_boards):
        print_numbers(i)

        # path to read
        path_file = path(size, i, amount_moves, num_dict)
        # read the file
        boards_game, amount_boards_of_game = read_file_to_list(path_file, size * size)

        if boards_game[amount_boards_of_game - 1]  in boards_game[:amount_boards_of_game-1]: # there is loop
            boards_game, amount_boards_of_game  = delete_repeat(boards_game, amount_boards_of_game)
        #after we delete the repeat boards, we split the board to series
        print(len(boards_game))
        splited_boards = split_boards(boards_game, amount_boards_of_game, IGNORE_RANGE, amount_board_in_series)
        print(len(splited_boards))
        series += splited_boards
    return series


def main():
    data = split_board_to_series(SIZE, AMOUNT_BOARDS, AMOUNT_MOVES, NUM_DICT, 3)
    # for i in data:
    #     print(i)
    print(len(data))


if __name__ == "__main__":
    main()
