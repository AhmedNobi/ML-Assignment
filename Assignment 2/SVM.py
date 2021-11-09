import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from assignment_functions import *

df = pd.read_csv('iris.csv')
target = df['variety']

rows = list(range(100,150))
df = df.drop(df.index[rows])

x = df['sepal.length']
x2 = df['petal.length']
 
setosa_x = x[:50]
setosa_x2 = x2[:50]

versicolor_x = x[50:]
versicolor_x2 = x2[50:]

plt.figure(figsize=(8,6))
plt.scatter(setosa_x,setosa_x2,marker='+',color='green')
plt.scatter(versicolor_x,versicolor_x2,marker='_',color='red')
plt.show()

## Drop rest of the features and extract the target values
df = df.drop(['sepal.width','petal.width'],axis=1)
Y = []
target = df['variety']
for val in target:
    if(val == 'Setosa'):
        Y.append(-1)
    else:
        Y.append(1)
df = df.drop(['variety'],axis=1)
X = df.values.tolist()
## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)

X = np.array(X)
Y = np.array(Y)

Y = Y.reshape(100,1)

y_pred,w = fit(X,Y)

## Predict

predictions = []
for val in y_pred:
    if(val >= 0):
        predictions.append(1)
    else:
        predictions.append(-1)

print(accuracy_score(Y,predictions))

#plot decision boundary
min_f1 = min(df['sepal.length'])
max_f1 = max(df['sepal.length'])
x_values = [(min_f1 - 1),(max_f1 + 1)]
y_values = - (w[0] + np.multiply(w[1], x_values)) / w[2]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.scatter(setosa_x,setosa_x2,marker='+',color='green')
plt.scatter(versicolor_x,versicolor_x2,marker='_',color='red')
plt.show()