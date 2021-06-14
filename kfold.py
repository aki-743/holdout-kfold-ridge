# ** K-分割交差検証 **

import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import ShuffleSplit

start = time.time()

iris = load_iris()

x = iris.data
y = iris.target

seed = 2021
test_items = np.linspace(0.1, 0.9, 9)

# for i in test_items:
#   n = 1
#   ss = ShuffleSplit(n_splits=3, test_size=i, random_state=seed)
#   for train_index, test_index in ss.split(x, y):
#     X_train = x[train_index]
#     y_train = y[train_index]
#     X_test = x[test_index]
#     y_test = y[test_index]

#     model = DTC(max_depth =3)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     score=accuracy_score(y_test, y_pred)
#     print("n=", n , "test_size=", '{:.2g}'.format(i), "----->", score)

#     n += 1

for j in range(1, 100, 1):
  for i in test_items:
    n = 1
    ss = ShuffleSplit(n_splits=3, test_size=i, random_state=seed)
    for train_index, test_index in ss.split(x, y):
      X_train = x[train_index]
      y_train = y[train_index]
      X_test = x[test_index]
      y_test = y[test_index]

      model = DTC(max_depth =3)
      model.fit(X_train, y_train)

      y_pred = model.predict(X_test)

      score=accuracy_score(y_test, y_pred)

      n += 1
  
print("処理時間:", time.time() - start)