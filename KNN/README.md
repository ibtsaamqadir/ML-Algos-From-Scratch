# K-Nearest Neighbors (KNN) Algorithm

This repository contains an implementation of the K-Nearest Neighbors (KNN) algorithm using both Euclidean and Manhattan distances. The main function `knn` is used to classify a test point based on the training dataset.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Overview](#algorithm-overview)
- [Function Parameters](#function-parameters)
- [Implementation](#implementation)

## Introduction

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm used for classification. It classifies a test point based on the majority class among its k-nearest neighbors from the training dataset. The distance between points can be measured using different metrics such as Euclidean or Manhattan distance.

## Algorithm Overview

The KNN algorithm works as follows:
1. **Compute Distance**: Calculate the distance between the test point and all points in the training dataset.
2. **Sort Neighbors**: Sort the training points based on the computed distance.
3. **Select Neighbors**: Select the k-nearest neighbors to the test point.
4. **Predict Class**: Determine the majority class among the k-nearest neighbors.

### Distance Metrics
- **Euclidean Distance**: The straight-line distance between two points in Euclidean space.
  
  d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
  
- **Manhattan Distance**: The sum of the absolute differences of their Cartesian coordinates.

  d(x, y) = \sum_{i=1}^{n} |x_i - y_i|


## Function Parameters

The function `knn` accepts the following parameters:
- **x_train**: Array of training dataset features.
- **y_train**: Array of training dataset labels.
- **x_test**: The test point, an array of the same size as the features in `x_train`.
- **k**: Integer, the number of neighbors to consider.
- **d_type**: Integer, type of distance metric (0 for Euclidean, 1 for Manhattan).

## Implementation

The main function is implemented as follows:

```python
def knn(x_train,y_train,x_test,k,d_type):
    m = len(x_train)
    dist = np.zeros(m)
    if d_type == 0: 
        for i in range(m):
            dist[i] = np.linalg.norm(x_test-x_train[i])
    elif d_type == 1:
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
        print(f'neighbour {l}:', y[l])
        nebor.append(y[l].astype(int))
    predicted_class = stat.mode(nebor)
    return predicted_class
```
