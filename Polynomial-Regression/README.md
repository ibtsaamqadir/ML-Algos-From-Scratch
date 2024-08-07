# Polynomial Regression Models

This repository contains implementations of 2nd-order (quadrativ) and 3rd-order polynomial regression models. The models compute the cost function and perform gradient descent to minimize the cost.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Implementation](#implementation)

## Introduction

This repo implements both 2nd-order and 3rd-order polynomial regression models to predict the profit of a food truck based on the population of a city. The models include the computation of the cost function and the use of gradient descent to optimize the model parameters. The vectorized approach is used to improve the speed of the computations.

## Data

The dataset `data.txt` consists of two columns:
- The first column represents the population of a city.
- The second column represents the profit of a food truck in that city, where a negative value indicates a loss.

## Implementation

The implementation includes the following steps:
1. **Loading Data**: Reading the data from `data.txt`.
2. **Polynomial Feature Expansion**: Creating polynomial features for 2nd- and 3rd-order regression.
3. **Cost Function**: Computing the cost function to measure the performance of the models.
4. **Gradient Descent**: Performing gradient descent to minimize the cost function and find the optimal parameters.

### nth-Order Polynomial Regression
- The nth-order (where n is 2 or 3) polynomial regression model includes an additional squared term to capture non-linear relationships between population and profit.
