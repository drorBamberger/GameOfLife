from functions import *

START = time.time()

stack = [generate_population(1, 100 * i) for i in range(AMOUNT_BOARDS)]
count = np.zeros(SIZE * SIZE + 1, dtype=int)
for i in stack:
    # print(i)
    count[sum(sum(i))] += 1

print(count)
print("the min index:", min_no_zero(count))
print("the max index:", max_no_zero(count))
print("the max value in index:", np.argmax(count), "with value:", max(count))
print("There are a", sum(count), "boards")

plt.plot(range(SIZE * SIZE + 1), count, 'bo', label='count')
plt.title('amount of alive pop')
plt.xlabel('amount of alive')
plt.ylabel('count in data')
plt.legend()

plt.show()

END = time.time()

print("the running take:", END - START, "sec")
