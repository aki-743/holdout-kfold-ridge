#   ** ホールドアウト法 **

import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC

start = time.time()

iris = load_iris()
x = iris.data
y = iris.target
seed = 2021

test_items = np.linspace(0.1, 0.9, 9)
# for i in test_items:
#   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i, random_state=seed)
#   model = DTC(max_depth = 3)
#   model.fit(X_train, y_train)

#   y_pred = model.predict(X_test)
#   score = accuracy_score(y_test, y_pred)
#   print("test_size=", '{:.2g}'.format(i), "----->", score)

# print("処理時間:", time.time() - start)

for j in range(1, 100, 1):
  for i in test_items:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i, random_state=seed)
    model = DTC(max_depth = 3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

print("処理時間:", time.time() - start)