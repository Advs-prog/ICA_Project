import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 8, 2000)
s1 = np.sin(2 * t)                 # source 1: smooth sine wave
s2 = np.sign(np.sin(3 * t))        # source 2: square wave
S = np.array([s1, s2])             # shape (2, 2000): one row per source

A = np.array([[1.0, 0.5],
              [0.6, 1.0]])         # mixing matrix (our choice)
X = A @ S                          # the two "microphone" recordings

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes[0, 0].plot(t, S[0]); axes[0, 0].set_title("Source 1 (sine)")
axes[1, 0].plot(t, S[1]); axes[1, 0].set_title("Source 2 (square)")
axes[0, 1].plot(t, X[0]); axes[0, 1].set_title("Mixed signal 1")
axes[1, 1].plot(t, X[1]); axes[1, 1].set_title("Mixed signal 2")
plt.tight_layout()
plt.savefig("toy_mixing.png", dpi=150)
plt.show()