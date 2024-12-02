import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Creating dataset
np.random.seed(0)
X_class0 = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], 50)
X_class1 = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 50)
X = np.vstack((X_class0, X_class1))
y = np.hstack((np.zeros(50), np.ones(50)))

# split the data 
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2)

# create an instance of the model
model = LogisticRegression()
model.fit(X_train, y_train)

# print the parameters
print('w:',model.coef_)
print('b:',model.intercept_)

# make predictions
y_pred = model.predict(X_test)
print('prediction for test data:',y_pred)
print('actual test data:',y_test)
# evaluate the model
accuracy = model.score(X_test, y_test)
print('accuracy:',accuracy)

# Step 4: Plot the data without decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression')
plt.show()

# Step 5: Plot the data with decision boundary
x1 = np.linspace(-1, 5, 100)
x2 = np.linspace(-1, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = model.predict(np.c_[X1.ravel(), X2.ravel()])
Z = Z.reshape(X1.shape)
plt.contourf(X1, X2, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression')
plt.show()





