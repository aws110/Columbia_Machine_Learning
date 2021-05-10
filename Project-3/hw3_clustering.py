import numpy as np
import sys

# X = np.genfromtxt(sys.argv[1], delimiter=",")

def KMeans(data):
    num_clusters = 5
    num_iterations = 10
    n = data.shape[1]
    centersList = np.random.random((5, n))

    for i in range(num_iterations):
        print("Iteration number KMeans " + str(i + 1))
        index = []
        for x in data:
            index.append(np.argmin(np.sum((centersList - x) ** 2, axis=1)))
        index = np.array(index)

        for j in range(num_clusters):
            count = np.sum(index == j)
            if count > 0:
                centersList[j] = np.mean(data[index == j], axis=0)
        filename = "centroids-" + str(i + 1) + ".csv"  # "i" would be each iteration
        np.savetxt(filename, centersList, delimiter=",")


def EMGMM(data):
    num_clusters = 5
    num_iterations = 10
    n = data.shape[1]
    mu = np.random.random((num_clusters, n))
    sigma = []
    for i in range(num_clusters):
        sigma.append(np.eye(n))
    sigma = np.array(sigma)
    pi = np.random.uniform(size=num_clusters)
    pi = pi / np.sum(pi)

    for i in range(num_iterations):
        print("Iteration number ENGMM " + str(i + 1))
        phis = []
        for x in data:
            phi = pi * np.array([np.linalg.det(np.linalg.inv(sigma[t])) * np.exp(-1 / 2 * (x - mu[t]).T.dot(np.linalg.inv(sigma[t])).dot(x - mu[t])) for t in range(num_clusters)])
            phi = phi / np.sum(phi)
            phis.append(phi)
        phis = np.array(phis)
        holder = np.sum(phis, axis=0)
        pi = holder / holder.sum()

        mu = np.transpose(phis).dot(data)
        mu != holder.reshape((-1, 1))

        for j in range(num_clusters):
            xi = data - mu[j]
            diag = np.diag(phis[:, j])
            s = xi.T.dot(diag).dot(xi)
            s /= holder[j]
            sigma[j] = s
            
        filename = "pi-" + str(i + 1) + ".csv"
        np.savetxt(filename, pi, delimiter=",")
        filename = "mu-" + str(i + 1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")  # this must be done at every iteration

        for j in range(num_clusters):
            filename = "Sigma-" + str(j + 1) + "-" + str(
                i + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, sigma[j], delimiter=",")



if __name__ == '__main__':
    data = np.genfromtxt(sys.argv[1], delimiter=",")

    KMeans(data)

    EMGMM(data)