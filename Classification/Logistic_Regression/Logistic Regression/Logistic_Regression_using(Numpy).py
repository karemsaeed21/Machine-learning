import numpy as np
import matplotlib.pyplot as plt
import copy

# Step 1 : Data Representation
np.random.seed(0)
X_class0 = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], 50)
X_class1 = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 50)
X = np.vstack((X_class0, X_class1))
y = np.hstack((np.zeros(50), np.ones(50)))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X,w,b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        cost += y[i] * np.log(f_wb_i) + (1 - y[i]) * np.log(1 - f_wb_i)
    cost = -cost / m
    return cost

def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i,j]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(X,y,w_in,b_in,alpha,num_iters):
    w  = copy.deepcopy(w_in)
    b = b_in
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b

w_init = np.zeros(X.shape[1])
b_init = 0.0
alpha = 0.01
num_iters = 5000

w , b = gradient_descent(X,y,w_init,b_init,alpha,num_iters)

print("w = ", w)
print("b = ", b)

# make prediction
X_test = np.array([[1, 1.5], [2, 1], [1.5, 1],[4.0,4]])
y_pred = predict(X_test,w,b)
print(y_pred)

for i in range(y_pred.shape[0]):
    if y_pred[i] > 0.5:
        print("class 1")
    else:
        print("class 0")


plt.scatter(X_class0[:,0], X_class0[:,1], color='red')
plt.scatter(X_class1[:,0], X_class1[:,1], color='blue')
x = np.linspace(-1, 5, 100)
y = -(w[0] * x + b) / w[1]
plt.plot(x, y, color='green')
plt.show()

