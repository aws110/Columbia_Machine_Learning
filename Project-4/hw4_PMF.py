from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter=",")

lam = 2
sigma2 = 0.1
d = 5
iterations = 50

def PMF(data):
    # initialize matrix
    L = np.zeros((iterations, 1))

    n_u = int(np.amax(data[:, 0]))
    u_matrices = np.zeros((iterations, n_u, d))

    n_v = int(np.amax(data[:, 1]))
    v_matrices = np.zeros((iterations, d, n_v))

    V = np.random.normal(0, 1 / lam, (d, n_v))

    # build M matrix
    M = np.zeros((n_u, n_v))
    missing_m = np.ones((n_u, n_v), dtype=np.int32)
    for i in data:
        r = int(i[0])
        c = int(i[1])
        M[r - 1, c - 1] = i[2]
        missing_m[r - 1, c - 1] = 0

    for i in range(iterations):
        v_matrices[i] = V
        U = np.zeros((n_u, d))

        # update U
        z1 = lam * sigma2 * np.eye(d)

        for j in range(n_u):
            z2 = np.zeros((d, d))
            z3 = np.zeros((d, 1))

            for k in range(n_v):
                if not missing_m[j, k]:
                    V_temp = V[:, k].reshape(d, 1)
                    z2 += np.dot(V_temp, np.transpose(V_temp))
                    z3 += M[j, k] * V_temp

            U[i, :] = np.dot(np.linalg.inv(z1 + z2), z3).reshape(-1)

        u_matrices[i] = U

        # calculate L
        p = 0
        for j in range(n_u):
            for k in range(n_v):
                if not missing_m[j, k]:
                    p += (M[j, k] - np.dot(U[j, :], np.transpose(V[:, k]))) ** 2
        p /= 2 * sigma2
        L[i] = - p - lam / 2 * (((np.linalg.norm(U, axis=1)) ** 2).sum()) - lam / 2 * (((np.linalg.norm(V, axis=0)) ** 2).sum())

        # update V
        z1 = lam * sigma2 * np.eye(d)
        V = np.zeros((d, n_v))

        for j in range(n_v):
            z2 = np.zeros((d, d))
            z3 = np.zeros((d, 1))

            for k in range(n_u):
                if not missing_m[k, j]:
                    U_temp = U[k, :].reshape(d, 1)
                    z2 += np.dot(U_temp, np.transpose(U_temp))
                    z3 += M[k, j] * U_temp

            V[:, j] = np.dot(np.linalg.inv(z1 + z2), z3).reshape(-1)

    return L, u_matrices, v_matrices

# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9].T, delimiter=",")
np.savetxt("V-25.csv", V_matrices[24].T, delimiter=",")
np.savetxt("V-50.csv", V_matrices[49].T, delimiter=",")

