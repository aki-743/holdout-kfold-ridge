# 正則化

import time

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

start = time.time()

iris = load_iris()

x = iris.data
y = iris.target
seed = 2021

test_items = np.linspace(0.1, 0.9, 9)
# for i in test_items:
#   X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i, random_state=seed)

#   # alphaが正則化のパラメータとなる
#   ridge = Ridge(alpha=0.01).fit(X_train, y_train)

#   score = ridge.score(X_test, y_test)
#   print("test_size=", '{:.2g}'.format(i), "----->", score)

for j in range(1, 100, 1):
  for i in test_items:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=i, random_state=seed)

    # alphaが正則化のパラメータとなる
    ridge = Ridge(alpha=0.01).fit(X_train, y_train)

    score = ridge.score(X_test, y_test)

print("処理時間:", time.time() - start)