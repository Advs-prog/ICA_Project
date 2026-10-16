import argparse
import glob
import os
import numpy as np
from scipy import signal
from audio_utils import load_audio

SR = 11025
N_FFT = 4096
HOP = 1024
MIN_QUERY_SECONDS = 3
LIBRARY_SECONDS = 360
AUDIO_TYPES = ("*.wav", "*.mp3", "*.m4a", "*.mp4")


def chroma_features(x):
    """Turn audio into normalized pitch-class features over time."""
    x = x - x.mean()
    freqs, _, spectrum = signal.stft(
        x, fs=SR, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP, nfft=N_FFT,
        boundary=None, padded=False
    )
    magnitude = np.abs(spectrum)
    chroma = np.zeros((12, magnitude.shape[1]))

    # Map frequency bins to one of the 12 musical pitch classes.
    useful = (freqs >= 55) & (freqs <= 2000)
    midi = np.rint(69 + 12 * np.log2(freqs[useful] / 440)).astype(int)
    pitch_classes = midi % 12
    for pitch in range(12):
        bins = pitch_classes == pitch
        if np.any(bins):
            chroma[pitch] = magnitude[useful][bins].sum(axis=0)

    # Normalize each frame so the comparison is less affected by volume.
    lengths = np.linalg.norm(chroma, axis=0)
    chroma /= np.maximum(lengths, 1e-12)
    return chroma


def compare_clip(query, song):
    """Find the best time-aligned match and return its score and offset."""
    query_frames = query.shape[1]
    if query_frames > song.shape[1]:
        return None, None

    scores = np.zeros(song.shape[1] - query_frames + 1)
    for pitch in range(12):
        scores += signal.correlate(
            song[pitch], query[pitch], mode="valid", method="fft"
        )

    best_frame = int(np.argmax(scores))
    score = scores[best_frame] / query_frames
    offset = best_frame * HOP / SR
    return float(score), offset


def find_songs(query_path, library_dir):
    """Rank library files by how closely they match a short query clip."""
    files = sorted(
        path
        for ext in AUDIO_TYPES
        for path in glob.glob(os.path.join(library_dir, ext))
    )
    if not files:
        raise SystemExit(f"No audio files found in '{library_dir}'")

    query_audio = load_audio(query_path, SR, seconds=10)
    if len(query_audio) < SR * MIN_QUERY_SECONDS:
        raise SystemExit("Use a query clip that is at least 3 seconds long")
    query = chroma_features(query_audio)

    results = []
    for path in files:
        try:
            song_audio = load_audio(path, SR, seconds=LIBRARY_SECONDS)
            song = chroma_features(song_audio)
            score, offset = compare_clip(query, song)
        except Exception as error:
            print(f"Skipping {os.path.basename(path)}: {error}")
            continue

        if score is not None:
            results.append((score, offset, os.path.basename(path)))

    if not results:
        raise SystemExit("No library songs were long enough to compare")

    return sorted(results, reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description="Find the closest song in a small local audio library."
    )
    parser.add_argument("query", help="Audio clip to identify (3 to 10 seconds)")
    parser.add_argument(
        "--library", default="library",
        help="Folder containing full song files (default: library/)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.library):
        raise SystemExit(f"Create the song library folder first: {args.library}")

    results = find_songs(args.query, args.library)
    print("Closest matches (score is a similarity, not a probability):")
    for rank, (score, offset, name) in enumerate(results[:3], start=1):
        print(f"{rank}. {name} — score {score:.3f}, around {offset:.1f}s into the song")

    if len(results) > 1 and results[0][0] - results[1][0] < 0.03:
        print("The top matches are close; try a longer or cleaner query clip.")


if __name__ == "__main__":
    main()
