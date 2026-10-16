import numpy as np


def center(X):
    return X - X.mean(axis=1, keepdims=True)


def whiten(Xc):
    eigvals, eigvecs = np.linalg.eigh(np.cov(Xc))
    return np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T @ Xc


def fastica(Z, n_components, rng, max_iter=200, tol=1e-6):
    W = np.zeros((n_components, Z.shape[0]))
    for i in range(n_components):
        w = rng.standard_normal(Z.shape[0])
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            g = np.tanh(w @ Z)
            w_new = (Z * g).mean(axis=1) - (1 - g ** 2).mean() * w
            w_new -= W[:i].T @ (W[:i] @ w_new)     # differ from earlier directions
            w_new /= np.linalg.norm(w_new)
            done = abs(abs(w_new @ w) - 1) < tol
            w = w_new
            if done:
                break
        W[i] = w
    return W @ Z


def best_match(S, est):
    """For each true source, find the recovered signal that correlates most."""
    N = len(S)
    C = np.corrcoef(S, est)[:N, N:]          # rows = true, columns = recovered
    idx = np.abs(C).argmax(axis=1)
    corr = C[np.arange(N), idx]              # signed correlation
    return idx, corr