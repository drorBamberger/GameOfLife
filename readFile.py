from PIL.Image import Resampling

from functions import *


def read_file(path, size, amount=1):
    length = size * amount
    if length % 8 != 0:
        length = length + 8 - length % 8

    # read file
    file_path = path_boards + path
    with open(file_path, 'rb') as f:
        binArray = f.read(length // 8)
    return binArray


def conv_bin_array_to_str(binArray, size, amount=1):
    # conv bin array to str
    res = ''
    for elem in binArray:
        res += bin(elem)[2:].zfill(8)

    length = size * amount
    # remove zeros
    if length % 8:
        res = res[:length]
    return res


def print_board(my_board, size, name):
    table = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            color = int(my_board[size * i + j]) * -1 + 1
            table[i, j] = [color, color, color]
    plt.imshow(table, interpolation='nearest')
    # plt.show()

    save(name, table)


def save(path, im):
    """
    Saves an image to file.

    If the image is type float, it will assume to have values in [0, 1].

    Parameters
    ----------
    path : str
        Path to which the image will be saved.
    im : ndarray (image)
        Image.
    """
    if im.dtype == np.uint8:
        pil_im = Image.fromarray(im)
    else:
        pil_im = Image.fromarray((im * 255).astype(np.uint8))
    fixed_height = 500
    resized_image = pil_im.resize((fixed_height, fixed_height), Resampling.NEAREST)
    resized_image.save(path)


AMOUNT = MOVES + 1
LEN = SIZE * SIZE
path_images = 'C:\\GameOfLife\\images\\'

if os.path.isdir(path_images[:-1]):
    shutil.rmtree(path_images[:-1])
os.mkdir(path_images[:-1])

bin_array = read_file('0\\10-2-6boards.bnr', LEN, AMOUNT)
boards = conv_bin_array_to_str(bin_array, LEN, AMOUNT)
print(boards)
for i in range(AMOUNT):
    board = boards[i * LEN:(i + 1) * LEN]
    print(board)
    print_board(board, SIZE, path_images + '10-2-' + str(i) + 'th6boards.png')
