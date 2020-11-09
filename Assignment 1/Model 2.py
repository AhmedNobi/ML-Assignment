import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
# Load data
data = pd.read_csv("house_data.csv")
sqft_living = data['sqft_living']
bathroom = data['bathrooms']
price = data['price']
# Set linear regression class
area = sqft_living + bathroom
area = np.expand_dims(area, axis=1)
price = np.expand_dims(price, axis=1)

linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(area, price)
prediction = linear_regression_model.predict(area)

plt.scatter(area, price)
plt.xlabel('bathrooms and internal area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.plot(area, prediction, color='red', linewidth=3)
plt.show()


print('Mean Square Error', metrics.mean_squared_error(price, prediction))