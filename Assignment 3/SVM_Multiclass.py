# importing necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.

y = iris.target

# Plot the samples
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0,shuffle=True)

# training a linear SVM classifier


svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_train, y_train)
#svm_predictions = svm_model_linear_ovr.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear_ovr.score(X_test, y_test)
print('One VS Rest SVM accuracy: ' + str(accuracy))
#
svm_model_linear_ovo = SVC(kernel='linear', C=1).fit(X_train, y_train)
#svm_predictions = svm_model_linear_ovo.predict(X_test)

# model accuracy for X_test
accuracy = svm_model_linear_ovo.score(X_test, y_test)
print('One VS One SVM accuracy: ' + str(accuracy))

h = .02  # step size in the mesh
#feature1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#feature2
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

#generate f1,f2 from min f1,f2 to max f1,f2 with step size h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

predictions = svm_model_linear_ovr.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
#change predictions to a meshgrid of the size of the inputs
predictions = predictions.reshape(xx.shape)
plt.contourf(xx, yy, predictions, cmap=plt.cm.coolwarm, alpha=0.45)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()