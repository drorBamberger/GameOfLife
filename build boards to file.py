from functions import *

START = time.time()

stack = generate_population(AMOUNT)

header = []
for i in range(SIZE):
    for j in range(SIZE):
        header.append('(' + str(i) + ',' + str(j) + ')')

data = []
count = np.zeros(SIZE * SIZE + 1, dtype=int)
for i in stack:
    data.append(i.flatten())
    count[sum(sum(i))] += 1

with open('C:\GameOfLife\data\FirstBoards.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(count)
print("There are a", sum(count), "boards")
END = time.time()
print("The running take:", END - START, "sec")
