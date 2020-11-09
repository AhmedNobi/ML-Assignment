import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
# Load data
data = pd.read_csv("house_data.csv")
sqft_living = data['sqft_living']
price = data['price']
# Set linear regression class
sqft_living = np.expand_dims(sqft_living, axis=1)
price = np.expand_dims(price, axis=1)

linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(sqft_living, price)
prediction = linear_regression_model.predict(sqft_living)

plt.scatter(sqft_living, price)
plt.xlabel('internal area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.plot(sqft_living, prediction, color='red', linewidth=3)
plt.show()

print('Mean Square Error', metrics.mean_squared_error(price, prediction))



