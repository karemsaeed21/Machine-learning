import numpy as np
import matplotlib.pyplot as plt

# Linear Regression: Custom Implementation
# Step 1: Data Representation
x_train = np.array([1, 2, 3, 4, 5])  # Population of cities (in 10,000s)
y_train = np.array([2, 4, 6, 8, 10])  # Profit (in 10,000s)

# Step 2: Hypothesis Function
def predict(x, w, b):
    return w * x + b

# Step 3: Cost Function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = predict(x[i], w, b)
        total_cost += (f_wb - y[i]) ** 2
    return total_cost / (2 * m)

# Step 4: Gradient Computation
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0 
    for i in range(m):
        f_wb = predict(x[i], w, b)
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    # Average the gradients over the number of examples
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

# Step 5: Gradient Descent Algorithm
def gradient_descent(x, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

# Step 6: Training the Model
w_init = 0
b_init = 0
alpha = 0.01
num_iters = 1000

w, b = gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters)

print(f"Trained parameters: w = {w}, b = {b}")

# Step 7: Making Predictions
x_new = np.array([5])
prediction = predict(x_new, w, b)
print(f"Prediction for input {x_new}: {prediction}")

# Step 8: Calculating Final Cost
print(f"Final cost after training: {compute_cost(x_train,y_train, w, b)}")

# # Step 9: Plot the Data and the Model
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, predict(x_train, w, b), color='red')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in 10,000s')
plt.title('Linear Regression Model')
plt.show()