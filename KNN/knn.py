import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

data = np.loadtxt('./C1_ex2data1.txt', delimiter = ',')
x, y = data[:,:2], data[:,-1]

def knn(x_train,y_train,x_test,k,d_type): #k is the number of neighnors and d_type is the type of distance
    m = len(x_train)
    dist = np.zeros(m)
    if d_type == 0: #eucledian distance 
        for i in range(m):
            dist[i] = np.linalg.norm(x_test-x_train[i])
    elif d_type == 1: #manhattan distance
        for i in range(m):
            dist[i] = np.sum(np.abs(x_test-x_train[i]))
    dist_1 = dist.copy()
    dist_1.sort()

    dis = np.zeros(k)
    nebor = []
    for j in range(k):
        index=np.where(dist==dist_1[j])
        dis[j] = index[0][0]
    for l in dis:
        l = l.astype(int)
        nebor.append(y[l].astype(int))
    predicted_class = stat.mode(nebor)
    
    return predicted_class

knn(x,y,[70,50],5,1) #test
