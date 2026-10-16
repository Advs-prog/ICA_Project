import os, glob
import numpy as np
import matplotlib.pyplot as plt
from audio_utils import load_audio, save_wav
from ica import center, whiten, fastica, best_match

SR = 22050
SECONDS = 10

files = sorted(f for ext in ("*.wav", "*.mp3", "*.m4a", "*.mp4")
               for f in glob.glob(os.path.join("sources", ext)))
N = len(files)
if N < 2:
    raise SystemExit("Put at least 2 audio files in the 'sources' folder")
names = [os.path.splitext(os.path.basename(f))[0] for f in files]
print("Sources:", names)

rng = np.random.default_rng(42)

# Load, trim to the shortest, equalize loudness
signals = [load_audio(f, SR, SECONDS) for f in files]
n = min(len(s) for s in signals)
S = np.array([s[:n] for s in signals])
S = S / np.abs(S).max(axis=1, keepdims=True)

# Mix, then separate
A = rng.random((N, N)) + np.eye(N)
X = A @ S
Z = whiten(center(X))            # PCA whitening
est = fastica(Z, N, rng)         # ICA

# Score ICA against the PCA baseline
idx_ica, corr_ica = best_match(S, est)
idx_pca, corr_pca = best_match(S, Z)
print(f"\n{'source':<14}{'PCA corr':>10}{'ICA corr':>10}")
for i, name in enumerate(names):
    print(f"{name:<14}{abs(corr_pca[i]):>10.3f}{abs(corr_ica[i]):>10.3f}")
if len(set(idx_ica)) < N:
    print("Warning: two sources matched the same output, separation failed")

# Save audio and plot
os.makedirs("audio", exist_ok=True)
os.makedirs("images", exist_ok=True)
sep = [est[idx_ica[i]] * np.sign(corr_ica[i]) for i in range(N)]
for i, name in enumerate(names):
    save_wav(f"audio/mixed_{i+1}.wav", X[i], SR)
    save_wav(f"audio/separated_{name}.wav", sep[i], SR)
for i, name in enumerate(names):
    save_wav(f"audio/pca_{name}.wav", Z[idx_pca[i]] * np.sign(corr_pca[i]), SR)

t = np.arange(n) / SR
fig, axes = plt.subplots(N, 3, figsize=(14, 2.3 * N), sharex=True, squeeze=False)
for i, name in enumerate(names):
    axes[i, 0].plot(t, S[i], linewidth=0.4);   axes[i, 0].set_title(f"Original: {name}")
    axes[i, 1].plot(t, X[i], linewidth=0.4);   axes[i, 1].set_title(f"Mixed {i+1}")
    axes[i, 2].plot(t, sep[i], linewidth=0.4); axes[i, 2].set_title(f"Separated: {name}")
plt.tight_layout()
plt.savefig("images/results.png", dpi=150)
plt.show()