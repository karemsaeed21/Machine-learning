import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Representation
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

def feature_normalize(X):
    """Normalize features to prevent numerical issues"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def predict(X, w, b):
    """Vectorized prediction"""
    return np.dot(X, w) + b

def compute_cost(X, y, w, b):
    """Vectorized cost computation"""
    m = X.shape[0]
    predictions = predict(X, w, b)
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    """Vectorized gradient computation"""
    m = X.shape[0]
    predictions = predict(X, w, b)
    error = predictions - y
    
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    """Gradient descent with history tracking and monitoring"""
    m = X.shape[0]
    cost_history = []
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Track progress
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return w, b, cost_history

# Training process
# First normalize the features
X_norm, mu, sigma = feature_normalize(X)

# Initialize parameters
w_init = np.zeros(X.shape[1])
b_init = 0
alpha = 0.01  # Increased learning rate since features are normalized
num_iters = 1000

# Train the model
w, b, cost_history = gradient_descent(X_norm, y, w_init, b_init, alpha, num_iters)

print(f"Trained parameters: w = {w}, b = {b}")

# Make predictions (remember to normalize new input)
X_new = np.array([2104, 5, 3, 15])
X_new_norm = (X_new - mu) / sigma
prediction = predict(X_new_norm, w, b)
print(f"Prediction for input {X_new}: ${prediction:.2f}k")

# Calculate final cost
final_cost = compute_cost(X_norm, y, w, b)
print(f"Final cost after training: {final_cost:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y, predict(X_norm, w, b), color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Perfect prediction line
plt.xlabel('Actual Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Plot cost history
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel('Iteration (x100)')
plt.ylabel('Cost')
plt.title('Cost History During Training')
plt.show()
