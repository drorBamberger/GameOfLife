from functions import *

START = time.time()
num_dict = 1
path_boards = 'C:\\GameOfLife\\boards'


def rand_bin(num, count):
    np.random.seed(num)
    res = b''
    for i in range(count):
        temp = bytes(random.randint(0, 1))
        res += temp
    return res


def create_board_to_file(num, size):
    initial_states = rand_bin(num, size * size)
    return initial_states


for i in range(num_dict):
    if not os.path.isdir(path_boards + " " + str(i)):
        os.mkdir(path_boards + " " + str(i))

for i in range(AMOUNT):
    name = str(SIZE) + "-" + str(i) + "FB"  # first board
    board = create_board_to_file(1, SIZE)
    with open(path_boards + " " + str(i % num_dict) + "\\" + name, 'wb') as f:
        f.write(board)

# stack = generate_population(AMOUNT)
#
#
# data = []
# with open('C:\GameOfLife\data\FirstBoards.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(data)
#
# count = np.zeros(SIZE * SIZE + 1, dtype=int)
# for i in stack:
#     data.append(i.flatten())
#     count[sum(i)] += 1
#
#
#
# print(count)
# print("There are a", sum(count), "boards")
# END = time.time()
# print("The running take:", END - START, "sec")
