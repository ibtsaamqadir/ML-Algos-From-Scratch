import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []

    for i in range(num_iters):
        h = np.dot(X, theta)
        errors = h - y  # Compute the errors for all training examples
        
        gradient = np.dot(X.T, errors) / m # Update theta for all parameters simultaneously
        theta = theta - alpha * gradient
        
        # Save the cost in every iteration
        J_history.append(computeCost(X, y, theta)) #computeCost() can be found in the Linear Regression or Polynomial Regression repo.
    
    return theta, J_history
