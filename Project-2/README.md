# Project 2 - Classification

Implementation a K-class Bayes classifier using maximum liklihood

To run code you need to enter:
`$ python3 hw2_classification.py X_train.csv y_train.csv X_test.csv`

The csv files that will be input into the code are formatted as follows:.

1. X_train.csv: A comma separated file containing the covariates. Each row corresponds to a single vector xi.
2. y_train.csv: A file containing the classes. Each row has a single number and the i-th row of this file combined with the i-th row of "X_train.csv" constitutes the labeled pair (yi,xi). There are 10 classes having index values 0,1,2,...,9.
3. X_test.csv: This file follows exactly the same format as "X_train.csv". No class file is given for the testing data.