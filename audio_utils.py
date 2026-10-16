import subprocess
import numpy as np
import imageio_ffmpeg
from scipy.io import wavfile


def load_audio(path, sr=22050, seconds=10):
    """Read any audio file (mp3, m4a, wav, mp4) as a mono array of numbers."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-v", "error", "-i", path, "-t", str(seconds),
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def save_wav(path, x, sr=22050):
    """Scale to a peak of 1 and write a WAV file."""
    wavfile.write(path, sr, (x / np.abs(x).max()).astype(np.float32))