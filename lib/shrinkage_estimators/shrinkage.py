import numpy as np


def shrinkage_cov_matrix(X, T):
    p, n = X.shape # Get shape of matrix

    x_mean = np.sum(X, axis=1)/n # Find mean of each row
    x_mean_M = np.tile(x_mean, (n, 1)).T # Repeat mean values as columns in a p x n matrix
    Y = X - x_mean_M

    W = []
    # Generate W array (which is 3D) of size (n, p, p), order of indices (k, i, j)
    for k in range(n):
        w = Y[:,k]
        W.append(np.outer(w, w))
    W_mean = np.sum(W, axis=0)/n

    # Emperically estimated covariance matrix
    S = n / (n-1) * W_mean

    W_mean_rep = np.tile(W_mean, (n, 1, 1))
    V = W - W_mean_rep
    # Compute variance of elements of the covariance matrix
    Var = n / (n-1)**3 * np.sum(V**2, axis=0)

    # Compute estimated shrinkage intensity parameter lambda
    lmbda_est = np.sum(Var) / np.sum((T-S)**2)

    # Restrict shrinkage intensity to interval [0,1]
    if lmbda_est < 0:
        lmbda_est = 0
    elif lmbda_est > 1:
        lmbda_est = 1

    # Compute shrinkage covariance matrix
    C_shrinkage = lmbda_est*T + (1-lmbda_est)*S

    return C_shrinkage, S, lmbda_est
