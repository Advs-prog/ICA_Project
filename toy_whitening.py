import numpy as np
import matplotlib.pyplot as plt

# Same toy setup as before
t = np.linspace(0, 8, 2000)
S = np.array([np.sin(2 * t), np.sign(np.sin(3 * t))])
A = np.array([[1.0, 0.5],
              [0.6, 1.0]])
X = A @ S

# Step 1: centering
X_c = X - X.mean(axis=1, keepdims=True)

# Step 2: whitening
C = np.cov(X_c)                          # covariance matrix
eigvals, eigvecs = np.linalg.eigh(C)     # D (eigvals) and E (eigvecs)
V = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
Z = V @ X_c                              # whitened data

print("Covariance before whitening:\n", np.cov(X_c).round(3))
print("Covariance after whitening:\n", np.cov(Z).round(3))

# Visual check
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X_c[0], X_c[1], s=3)
axes[0].set_title("Centered (stretched, correlated)")
axes[1].scatter(Z[0], Z[1], s=3)
axes[1].set_title("Whitened (uncorrelated, equal spread)")
for ax in axes:
    ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("toy_whitening.png", dpi=150)
plt.show()