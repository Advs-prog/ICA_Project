import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# --- Setup (same as before) ---
t = np.linspace(0, 8, 2000)
S = np.array([np.sin(2 * t), np.sign(np.sin(3 * t))])
A = np.array([[1.0, 0.5],
              [0.6, 1.0]])
X = A @ S

# --- Center and whiten ---
X_c = X - X.mean(axis=1, keepdims=True)
eigvals, eigvecs = np.linalg.eigh(np.cov(X_c))
V = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
Z = V @ X_c

# --- FastICA (finds one direction at a time) ---
def fastica(Z, n_components, max_iter=200, tol=1e-6):
    n = Z.shape[0]
    W = np.zeros((n_components, n))
    for i in range(n_components):
        w = np.random.randn(n)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            wz = w @ Z                       # projection of data on w
            g = np.tanh(wz)                  # non-Gaussianity measure
            g_prime = 1 - g ** 2
            w_new = (Z * g).mean(axis=1) - g_prime.mean() * w
            w_new -= W[:i].T @ (W[:i] @ w_new)   # stay different from earlier directions
            w_new /= np.linalg.norm(w_new)
            done = abs(abs(w_new @ w) - 1) < tol
            w = w_new
            if done:
                break
        W[i] = w
    return W

W = fastica(Z, n_components=2)
S_est = W @ Z                                # recovered sources

# --- Compare with originals ---
corr = np.corrcoef(S, S_est)[:2, 2:]
print("Correlation (rows = true sources, columns = recovered):\n", corr.round(3))

fig, axes = plt.subplots(2, 3, figsize=(12, 5))
for i in range(2):
    axes[i, 0].plot(t, S[i]);      axes[i, 0].set_title(f"True source {i+1}")
    axes[i, 1].plot(t, X[i]);      axes[i, 1].set_title(f"Mixed signal {i+1}")
    axes[i, 2].plot(t, S_est[i]);  axes[i, 2].set_title(f"Recovered {i+1}")
plt.tight_layout()
plt.savefig("images/toy_ica.png", dpi=150)
plt.show()