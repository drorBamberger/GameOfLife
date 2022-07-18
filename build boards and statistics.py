import numpy as np
import time
import matplotlib.pyplot as plt

start = time.time()
SIZE = 20

M = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


# the calcUneighs, make_move and generate_population functions from:
# https://medium.com/@ptyshevs/rgol-ga-1cafc67db6c7

def calc_neighs(field, i, j):
    """ Calculate number of neighbors alive (assuming square field) """
    neighs = 0
    n = len(field)
    for m in M:
        row_idx = m[0] + i
        col_idx = m[1] + j
        if 0 <= row_idx < n and 0 <= col_idx < n:
            if field[row_idx][col_idx]:
                neighs += 1
    return neighs


def make_move(field, moves=1):
    """ Make a move forward according to Game of Life rules """
    n = len(field)
    cur_field = field
    for _ in range(moves):
        new_field = np.zeros((n, n), dtype='uint8')
        for i in range(n):
            for j in range(n):
                neighs = calc_neighs(cur_field, i, j)
                if cur_field[i][j] and neighs == 2:
                    new_field[i][j] = 1
                if neighs == 3:
                    new_field[i][j] = 1
        cur_field = new_field
    return cur_field


def generate_population(amount, random_state=-1):
    """
    Generating initial population of individual solutions
    :return: initial population as a list of 20x20 arrays
    """
    if random_state != -1:
        np.random.seed(random_state)
    initial_states = np.split(np.random.binomial(1, 0.5, (SIZE * amount, SIZE)).astype('uint8'), amount)
    return deduction_edges(make_move(state, 0) for state in initial_states)


def deduction_edges(l):
    for i in l:
        if sum(sum(i)) < SIZE * SIZE * 0.05:
            return []
        elif sum(sum(i)) > SIZE * SIZE * 0.95:
            return []
        else:
            return i


def Average(l):
    avg = sum(l) / len(l)
    return avg


def min_no_zero(arr):
    for i in range(len(arr)):
        if arr[i] != 0:
            return i
    return -1


def max_no_zero(arr):
    for i in range(len(arr)):
        if arr[-i] != 0:
            return len(arr) - i
    return -1


amount = 1000000

# stack,CL,CH = generate_population(amount)

stack = []
for i in range(amount):
    # print(generate_population(1, i))
    stack.append(generate_population(1, 100 * i))

count = np.zeros(SIZE * SIZE + 1, dtype=int)
for i in stack:
    # print(i)
    count[sum(sum(i))] += 1

print(count)
print("the min index:", min_no_zero(count))
print("the max index:", max_no_zero(count))
print("the max value in index:", np.argmax(count), "with value:", max(count))
print("There are a", sum(count), "boards")

plt.plot(range(0, SIZE * SIZE + 1), count, 'bo', label='count')
plt.title('amount of alive pop')
plt.xlabel('amount of alive')
plt.ylabel('count in data')
plt.legend()

plt.show()

end = time.time()

print("the running take:", end - start, "sec")
