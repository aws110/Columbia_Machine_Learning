import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file

    d = X_train.shape[1] #dimension
    X = np.matrix(X_train)
    X_transpose = np.transpose(X)
    y = np.matrix(y_train)
    y = np.transpose(y)
    I = np.identity(d) #identity matrix of dimension x dimension

    return np.linalg.inv(lambda_input * I + X_transpose * X) * (X_transpose * y)

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file

    d = X_train.shape[1]  # dimension
    X = np.matrix(X_train)
    X_transpose = np.transpose(X)
    I = np.identity(d)  # identity matrix of dimension x dimension
    X_t = np.matrix(X_test)
    indices = list(range(1, len(X_t) + 1))

    covariance = np.linalg.inv(lambda_input * I + sigma2_input ** (-1) * X_transpose * X)

    obj = []
    while len(obj) <= 10:
        vec = []
        for i in range(len(X_t)):
            cov_temp = np.linalg.inv(np.linalg.inv(covariance) + sigma2_input ** (-1) * np.transpose(X_t[i]) * X_t[i])

            vec.append(sigma2_input + X_test[i] * cov_temp * np.transpose(X_t[i]))

        index = np.argmax(vec)

        covariance = np.linalg.inv(np.linalg.inv(covariance) + sigma2_input ** (-1) * np.transpose(X_t[index]) * X_t[index])

        X_t = np.delete(X_t, index, 0)

        obj.append(indices[index])
        indices.remove(indices[index])

    return obj

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file