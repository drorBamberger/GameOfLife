from functions import *

boards = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
length = 4
ignore_range = 5
amount_board_in_series = 5

boards = boards[ignore_range:]
print([(boards[i:i + amount_board_in_series]) for i in range(len(boards) - amount_board_in_series + 1)])

