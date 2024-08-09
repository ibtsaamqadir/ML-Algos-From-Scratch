import numpy as np
import matplotlib.pyplot as plt
import math

data = np.loadtxt('./data.txt', delimiter = ',')
x_train, y_train = data[:,:2], data[:,-1]

def sigmoid(z):
    result = 1/(1+np.exp(-z))
    
    return result

def computeCost(X, y, w, b, lambda_= 1):
    m, n = X.shape
    J = 0 
    for i in range(m):                   #training example
        z_wb = 0 
        for j in range(n):               #features
             z_wb_ij =  w[j]*X[i][j]
             z_wb += z_wb_ij             # z_wb = z_wb + z_wb_ij
        z_wb += b                        # z_wb = z_wb + b
        f_wb = sigmoid(z_wb)
        J += (-y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)) #log loss 
    total_cost = (1 / m) * J

    return total_cost

def compute_gradient(X, y, w, b, lambda_=None): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = 0
        for j in range(n): 
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij
        z_wb += b
        f_wb = sigmoid(z_wb)
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw_ij = (f_wb - y[i])* X[i][j] 
            dj_dw[j] += dj_dw_ij 
            
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    m = len(X)
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

    return w_in, b_in, J_history

np.random.seed(1)
init_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
init_b = -8

iterations = 10000
alpha = 0.001

w,b, J_history = gradient_descent(x_train ,y_train, init_w, init_b, 
                                   computeCost, compute_gradient, alpha, iterations, 0)

def predict(X, w, b): 
    m, n = X.shape   
    p = np.zeros(m)
    for i in range(m):   
        z_wb = 0
        for j in range(n): 
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij
        z_wb += b
        
        f_wb = sigmoid(z_wb)
        p[i] = f_wb >= 0.5
    return p

p = predict(x_train, w,b)
print('Train Accuracy: ',(np.mean(p == y_train) * 100))
