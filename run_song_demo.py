import argparse
import glob
import itertools
import json
import os
import re
import numpy as np
from audio_utils import load_audio, save_wav
from ica import center, whiten, fastica

SR = 11025
CLIP_SECONDS = 8
LIBRARY_SECONDS = 300
DEFAULT_COUNT = 2
AUDIO_TYPES = ("*.wav", "*.mp3", "*.m4a", "*.mp4")
OUTPUT_DIR = "song_demo_output"


def find_audio_files(folder):
    """Return supported audio files directly inside a folder."""
    files = []
    for ext in AUDIO_TYPES:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def random_clip(path, rng):
    """Load a song and choose a random eight-second excerpt."""
    audio = load_audio(path, SR, seconds=LIBRARY_SECONDS)
    clip_length = SR * CLIP_SECONDS
    if len(audio) < clip_length:
        raise ValueError(f"song is shorter than {CLIP_SECONDS} seconds")

    start = int(rng.integers(0, len(audio) - clip_length + 1))
    clip = audio[start:start + clip_length]
    peak = np.abs(clip).max()
    if peak < 1e-12:
        raise ValueError("song excerpt is silent")
    return clip / peak, start / SR


def choose_songs(folder, count, rng):
    """Randomly choose usable songs and one excerpt from each."""
    files = find_audio_files(folder)
    if len(files) < 2:
        raise SystemExit(f"Add at least two audio files to '{folder}'")

    rng.shuffle(files)
    sources = []
    for path in files:
        try:
            clip, start = random_clip(path, rng)
        except Exception as error:
            print(f"Skipping {os.path.basename(path)}: {error}")
            continue

        sources.append({
            "name": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "clip": clip,
            "start": start,
        })
        if len(sources) == count:
            break

    if len(sources) < count:
        raise SystemExit(f"Need {count} usable songs in '{folder}' for this run")
    return sources


def match_outputs(sources, estimates):
    """Assign each separated signal to a different original song by correlation."""
    source_signals = np.array([item["clip"] for item in sources])
    n = len(sources)
    correlations = np.corrcoef(source_signals, estimates)[:n, n:]

    # ICA output order and sign are arbitrary, so select the best one-to-one match.
    assignment = max(
        itertools.permutations(range(n)),
        key=lambda order: sum(abs(correlations[i, order[i]]) for i in range(n)),
    )

    matches = []
    for source_index, output_index in enumerate(assignment):
        matches.append({
            "source_index": source_index,
            "output_index": output_index,
            "correlation": float(correlations[source_index, output_index]),
        })
    return matches


def safe_name(name):
    """Make a short filename from a library song name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return cleaned or "song"


def run_demo(library_dir, count, seed):
    rng = np.random.default_rng(seed)
    sources = choose_songs(library_dir, count, rng)
    n = len(sources)
    S = np.array([item["clip"] for item in sources])

    # Each row of X is a mixed recording made from the selected songs.
    A = rng.random((n, n)) + np.eye(n)
    X = A @ S

    # Reuse the same whitening and FastICA functions as the audio ICA demo.
    Z = whiten(center(X))
    estimates = fastica(Z, n, rng)
    matches = match_outputs(sources, estimates)

    selected_dir = os.path.join(OUTPUT_DIR, "selected_songs")
    mixed_dir = os.path.join(OUTPUT_DIR, "mixed_songs")
    separated_dir = os.path.join(OUTPUT_DIR, "separated_songs")
    for folder in (selected_dir, mixed_dir, separated_dir):
        os.makedirs(folder, exist_ok=True)

    source_rows = []
    for i, item in enumerate(sources):
        filename = f"song_{i + 1}_{safe_name(item['name'])}.wav"
        relative_path = os.path.join("song_demo_output", "selected_songs", filename).replace("\\", "/")
        save_wav(relative_path, item["clip"], SR)
        source_rows.append({
            "name": item["name"],
            "start_seconds": round(item["start"], 1),
            "file": relative_path,
        })

    mixed_rows = []
    for i in range(n):
        filename = f"mixture_{i + 1}.wav"
        relative_path = os.path.join("song_demo_output", "mixed_songs", filename).replace("\\", "/")
        save_wav(relative_path, X[i], SR)
        mixed_rows.append({"name": f"Mixed recording {i + 1}", "file": relative_path})

    separated_rows = []
    for match in matches:
        source_index = match["source_index"]
        output_index = match["output_index"]
        source_name = sources[source_index]["name"]
        filename = f"separated_{source_index + 1}_{safe_name(source_name)}.wav"
        relative_path = os.path.join("song_demo_output", "separated_songs", filename).replace("\\", "/")
        sign = 1 if match["correlation"] >= 0 else -1
        save_wav(relative_path, estimates[output_index] * sign, SR)
        separated_rows.append({
            "name": f"Separated output {output_index + 1}",
            "matched_song": source_name,
            "correlation": round(abs(match["correlation"]), 3),
            "file": relative_path,
        })

    data = {
        "clip_seconds": CLIP_SECONDS,
        "sample_rate": SR,
        "sources": source_rows,
        "mixing_matrix": np.round(A, 3).tolist(),
        "mixtures": mixed_rows,
        "separated": separated_rows,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"Picked {n} songs, mixed them, whitened the mixtures, and ran FastICA.")
    for item in separated_rows:
        print(f"{item['name']} matched to {item['matched_song']} "
              f"(correlation {item['correlation']:.3f})")
    print(f"Separated WAV files: {separated_dir}")
    print("Open song_demo.html through a local web server to view this run.")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly mix two songs, separate them with ICA, and match outputs."
    )
    parser.add_argument("--library", default="library", help="Folder with song files")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for a repeatable demo")
    args = parser.parse_args()

    if not os.path.isdir(args.library):
        raise SystemExit(f"Create the song library folder first: {args.library}")
    run_demo(args.library, DEFAULT_COUNT, args.seed)


if __name__ == "__main__":
    main()

