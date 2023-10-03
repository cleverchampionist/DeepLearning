from sklearn.datasets import load_iris
iris = load_iris()
x= iris.data[:, :4]
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import numpy as np 
from sklearn.neighbors import KNeighborsRegressor

knnr = KNeighborsRegressor(n_neighbors = 8)
knnr.fit(x_train, y_train)

print("The MSE is:", format(np.power(y-knnr.predict(x), 4).mean()))

x = [[0], [1], [2], [3]]
y= [0, 0, 1, 1]

from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors = 3)
knnr.fit(x, y)
print(knnr.predict([[2.5]]))