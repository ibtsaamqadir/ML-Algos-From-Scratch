# Logistic Regression

This repository contains an implementation of Logistic Regression, a fundamental classification algorithm in machine learning. The data used for this implementation is based on two features for two classes. The model employs a linear decision boundary, log loss function, and sigmoid activation function.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Overview](#algorithm-overview)
- [Implementation](#implementation)

## Introduction

Logistic Regression is a common classification algorithm used to predict the probability of a binary outcome. Here it is used for binary classification problems where the output can be one of two possible classes.

## Algorithm Overview

The Logistic Regression algorithm works as follows:

1. **Sigmoid Function**: The sigmoid function is used to map the predicted values to probabilities. It is defined as:
    sigmoid(z) = 1/{1 + e^{-z}}
2. **Decision Boundary**: A linear decision boundary separates the two classes.
3. **Cost Function (Log Loss)**: The log loss function, also known as binary cross-entropy, measures the performance of a classification model whose output is a probability value between 0 and 1.
4. **Gradient Descent**: An optimization algorithm used to minimize the cost function by iteratively updating the model parameters.

## Implementation

The implementation includes the following steps:
1. **Loading Data**: Reading the data from a .txt file.
2. **Visualization**: Plotting the data to understand the relationship between the features and the target variable.
3. **Sigmoid Function**: Implementing the sigmoid function to map predictions to probabilities.
4. **Cost Function**: Implementing the log loss function to measure the performance of the model.
5. **Gradient Descent**: Performing gradient descent to minimize the cost function and find the optimal parameters.
6. **Prediction**: Using the trained model to make predictions on new data.
