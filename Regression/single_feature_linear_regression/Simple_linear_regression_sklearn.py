import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Step 1: Data Representation
x = np.array([1, 2, 3, 4, 5 , 6 , 7, 8 ,9 ,10]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10 ,12,14,16,18,20]).reshape(-1, 1)

# plt.scatter(x, y, color='blue')
# plt.title('Population of cities vs Profit')
# plt.xlabel('Population of cities (in 10,000s)')
# plt.ylabel('Profit (in 10,000s)')
# plt.show()


# Step 2: Initialize the Linear Regression Model
model = LinearRegression()

# split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Step 3: Train the Model
model.fit(x_train, y_train)


# Step 4: Extract the Parameters (w and b)
print(f"Trained parameters: w = {model.coef_[0] }, b = {model.intercept_}")

# Step 5: Making Predictions
prediction = model.predict(x_test)
print(f"Prediction for input {x_test}: {prediction}")

# Step 6: Calculating Final Cost
cost = mean_squared_error(y_test, prediction)
print(f"Final cost after training: {cost}")

# score
print(f"Score: {model.score(x_test, y_test)}")

# Step 7: Plot the Data and the Model
plt.scatter(x, y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.title('Population of cities vs Profit')
plt.xlabel('Population of cities (in 10,000s)')
plt.ylabel('Profit (in 10,000s)')
plt.show()
