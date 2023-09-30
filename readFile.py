from PIL.Image import Dither
from functions import *


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
    boards, amount_boards = read_file_to_list(PATH_TO_READ, LEN)
    i = 0
    # save the boards to images file
    for board in boards:
        path_image_file = PATH_IMAGES + str(SIZE) + '-' + str(READ_FILE) + '-' + str(i) + 'th' + str(
            AMOUNT_MOVES) + 'boards.png'
        save_board(board, SIZE, path_image_file)
        i += 1
    print(i)


if __name__ == "__main__":
    main()
