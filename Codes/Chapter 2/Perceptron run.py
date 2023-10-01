import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = (r'C:\Users\USUARIO\Desktop\Personal\Cursos y lectures\Python Machine Learning - Raschka\Datasets\iris.txt')
# print('Archivo:', s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')

# print(df.tail())

# Select setosa and versicolor

Y = df.iloc[0:100, 4].values
Y = np.where(Y=='Iris-setosa', -1, 1)

# Extract sepal length and petal length
X = df.iloc[0:100, [0,2]].values

# plot data
plt.scatter(X[:50,0], X[:50, 1], color='red', marker='o', label='setosa')
# print(X)