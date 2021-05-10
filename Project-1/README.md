# Project 1 - Ridge Regression

## Part 1:
In this part you will implement the ℓ2-regularized least squares linear regression algorithm.

## Part 2:
In the same code, you will also implement the active learning procedure

To execute the code run the following code:
`$ python3 hw1_regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv`

The csv files to input into your code are formatted as follows:.

1. X_train.csv: A comma separated file containing the covariates. Each row corresponds to a single vector xi. The last dimension has already been set equal to 1 for all data.
2. y_train.csv: A file containing the outputs. Each row has a single number and the i-th row of this file combined with the i-th row of "X_train.csv" constitutes the training pair (yi,xi).
3. X_test.csv: This file follows exactly the same format as "X_train.csv". No response file is given for the testing data.