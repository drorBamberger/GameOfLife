

# sourcery skip: use-itertools-product
from sklearn import tree
from functions import *
from split_to_series import *

data = []
X = []

y = [[int(line[-1][i]) for line in data] for i in range(LEN)]

X_train = []
X_test = []
y_train = []
y_test = []

for j in range(len(X)):
        if j%10 != 0:
            X_train.append(X[j])
        else:
            X_test.append(X[j])

for i in range(LEN):
    temp_train = []
    temp_test = []
    for j in range(len(X)):
        if j%10 != 0:
            temp_train.append(y[i][j])
        else:
            temp_test.append(y[i][j])
    y_train += temp_train
    y_test += temp_test


dts = []
for i in range(LEN):
    dt = tree.DecisionTreeClassifier(max_depth=11, random_state=42)
    dt = dt.fit(X_train, y_train[i])
    dts.append(dt)

# The error on the training and test data sets
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                                measure_error(y_test, y_test_pred, 'test')],
                                axis=1)

print(train_test_full_error)
            
