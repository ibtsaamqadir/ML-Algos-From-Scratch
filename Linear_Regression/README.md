# Linear Regression Model

This repository contains an implementation of a very simple Linear Regression model. The model computes the cost function and performs gradient descent to minimize the cost. The dataset used (`data.txt`) contains two columns: the population of a city and the profit of a food truck in that city. A negative value for profit indicates a loss.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Implementation](#implementation)

## Introduction

This project implements a simple Linear Regression model to predict the profit of a food truck based on the population of a city. The model includes the computation of the cost function and the use of gradient descent to optimize the model parameters.

## Data

The dataset `data.txt` consists of two columns:
- The first column represents the population of a city.
- The second column represents the profit of a food truck in that city, where a negative value indicates a loss.

## Implementation

The implementation includes the following steps:
1. **Loading Data**: Reading the data from `data.txt`.
2. **Visualization**: Plotting the data to understand the relationship between population and profit.
3. **Cost Function**: Computing the cost function to measure the performance of the model.
4. **Gradient Descent**: Performing gradient descent to minimize the cost function and find the optimal parameters.
