import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
X = np.array([
    [2104, 5, 3, 15],
    [1416, 3, 2, 40],
    [852, 2, 1, 35],
    [1800, 4, 2, 20],
    [2200, 5, 3, 10],
    [1100, 2, 1, 50],
    [1700, 3, 2, 25],
    [2500, 6, 4, 5],
    [1300, 3, 1, 30],
    [1950, 4, 3, 18],
    [2000, 4, 2, 22],
    [1600, 3, 2, 28],
    [2400, 5, 3, 12],
    [1200, 2, 2, 45],
    [1800, 4, 2, 32]
])

# Target: House prices in $1000s
y = np.array([400, 232, 178, 330, 450, 190, 310, 520, 240, 380, 360, 290, 470, 220, 340])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size= 0.2, random_state=42)

# # Step 1: Data Preprocessing
# # Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# # Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Step 2: Model Training
model = LinearRegression()

# Step 3: Model Evaluation
# Train the model
model.fit(X_train, y_train)

print(f"Model coefficients: {model.coef_}")
print(f"Model intercept: {model.intercept_}")

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")

# Step 4: Plotting the Data
# Create a scatter plot of actual vs predicted values instead
plt.figure(figsize=(10, 6))
plt.scatter(y_train, model.predict(X_train), color='blue', label='Training data')
plt.scatter(y_test, y_pred, color='red', label='Test data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Perfect prediction line
plt.xlabel('Actual Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()
