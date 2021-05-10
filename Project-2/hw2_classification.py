from __future__ import division
import numpy as np
import sys
from scipy.stats import multivariate_normal

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")


def pluginClassifier(X_train, y_train, X_test):
    y_train_set = set(y_train)
    n = len(y_train_set)

    priors = []
    for i in y_train_set:
        vals = y_train.tolist().count(i) / n
        priors.append(vals)

    #empty lists for mean, covariance, and conditionals
    mean = []
    cov = []
    conditional = []
    for i in y_train_set:
        #Calculate means for each X
        mean.append(np.mean(X_train[np.where(y_train == i)[0]], axis=0))
        #Calculate covariance for each X
        cov.append(np.cov(X_train[np.where(y_train == i)[0]], rowvar=0))

        conditional.append(multivariate_normal(mean=mean[-1], cov=cov[-1]))

    #Empty list for posteriors
    posteriors = []
    for i in X_test:

        numerator = []
        for j in range(n):
            val = priors[j] * conditional[j].pdf(i) #probability density function
            numerator.append(val)

        predictions = []
        for j in range(n):
            prediction = numerator[j] / sum(numerator)
            predictions.append(prediction)

        posteriors.append(predictions)

    return np.array(posteriors)


final_outputs = pluginClassifier(X_train, y_train, X_test)  # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",")  # write output to file