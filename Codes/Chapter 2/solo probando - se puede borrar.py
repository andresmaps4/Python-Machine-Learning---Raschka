import numpy as np
import pandas as pd

matriz_1 = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
matriz_2 = [[13,14],[3,8],[17,9]]
X = pd.DataFrame(matriz_1)
Y = pd.DataFrame(matriz_2)

a = np.array(matriz_1)
b = np.array(matriz_2)

rgen = np.random.RandomState()
w_ = rgen.normal(loc=0.0, scale=0.01,
                size=1 + X.shape[1])

# print(X)

# print(X.shape)

# print(X.shape[1])
# print(w_[0])

# print(w_)

producto_0 = np.dot(X, w_[1:]) 
producto_1 = np.dot(X, w_[1:]) + w_[0]

# print(producto_0)

# print(producto_1)

cosa_1 = np.where(producto_1 >= 0.0, 1, -1)

# print('predicción:', cosa_1)

cosa = sum([i*j for i,j in zip(a,b.T)])

cosa_1 = [(i,j) for i,j in zip(a,b.T)]

# cosa_2 = zip(X,Y)

# print(cosa)

# print(a)
# print(b)

# cosa_3 = np.dot(a,b)

print(cosa_1)

print(cosa)

## Acá es para probar algunas cositas ###

import numpy as np
import pandas as pd

s = (r'C:\Users\USUARIO\Desktop\Personal\Cursos y lectures\Python Machine Learning - Raschka\Datasets\iris.txt')

df = pd.read_csv(s, header = None, encoding = "utf-8")

random_state = 1
rgen = np.random.RandomState(random_state)
X = df.iloc[0:100, [0,2]].values
y = df.iloc[0:100, 4].values
r = rgen.permutation(len(y))
print(r)
print(X[r])

X = np.array([[1,2,3], [4,5,6]])
print(np.ravel(X).shape[0])