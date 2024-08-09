import numpy as np
import matplotlib.pyplot as plt

path = "./data.txt"

# Read comma-separated data
data = np.loadtxt(path, delimiter=',')
X, Y = data[:, 0], data[:, 1]

m = Y.size  # number of training examples
X = np.stack([np.ones(m), X], axis=1) # it used to convert X in to (97x2), first colum is all ones to get  where theta is (2x1)    "theta[0]+theta[1]*X"

def computeCost(X,y , theta):
  m = y.size
  J = 0
  h = np.dot(X, theta)
  for i in range(m):
      J = J + (h[i] - y[i])**2
  J = J/(2*m)
  return J

J = computeCost(X, Y, theta=np.array([0.0, 0.0]))

def gradientDescent(X, y, theta, alpha, num_iters):
  m = y.shape[0]
  theta = theta.copy()
  J_history = []

  for i in range(num_iters):
      for j in range(m):
          h = np.dot(X, theta)
          theta[0] =  theta[0] - (1/m)*alpha*((h[j] - y[j])*X[j,0])
          theta[1] =  theta[1] - (1/m)*alpha*((h[j] - y[j])*X[j,1])
      J_history.append(computeCost(X, y, theta))
  return theta, J_history

theta = np.zeros(2)
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,Y, theta, alpha, iterations)

plt.scatter(X[:, 1],Y, marker = 'x', color = 'red')
plt.plot(X[:, 1], np.dot(X, theta))
plt.legend(['Training data',  'Linear regression'])
