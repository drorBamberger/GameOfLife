from PIL.Image import Dither
from functions import *


def read_file_bin_array(path, size, amount=1):
    """the function read the bin file, and insert the board to bin array"""
    length = size * amount
    if length % BYTE != 0:
        length = length + BYTE - length % BYTE

    # read file
    file_path = PATH_BOARDS + path
    with open(file_path, 'rb') as f:
        binArray = f.read(length // 8)
    return binArray


def read_file_string(path, len, amount=1):
    """the function read the bin file, and insert the board to str array"""

    bin_array = read_file_bin_array(path, len, amount)
    list_boards = conv_bin_array_to_str(bin_array, len, AMOUNT_GENERATIONS)
    return list_boards


def conv_bin_array_to_str(binArray, size, amount=1):
    """the function convert bin array to string"""

    # conv bin array to str
    res = ''
    for elem in binArray:
        res += bin(elem)[2:].zfill(BYTE)

    length = size * amount
    # remove zeros
    if length % BYTE:
        res = res[:length]
    return res


def save_board(my_board, size, name):
    """the function plot the board, and save the board to image file"""
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
    resized_image = pil_im.resize((fixed_height, fixed_height), Dither.NONE)
    resized_image.save(path)


def main():
    # Delete the previous images
    if os.path.isdir(PATH_IMAGES[:-1]):
        shutil.rmtree(PATH_IMAGES[:-1])
    os.mkdir(PATH_IMAGES[:-1])

    # read the file to string
    boards = read_file_string(PATH_TO_READ, LEN, AMOUNT_GENERATIONS)

    # save the boards to images file
    for i in range(AMOUNT_GENERATIONS):
        board = boards[i * LEN:(i + 1) * LEN]
        path_image_file = PATH_IMAGES + str(SIZE) + '-' + str(READFILE) + '-' + str(i) + 'th' + str(
            AMOUNT_GENERATIONS) + 'boards.png'
        save_board(board, SIZE, path_image_file)


if __name__ == "__main__":
    main()
