"""Reading sound and measuring how loud it is. Every lane that looks at levels uses these."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clipper import shell

SILENCE_DB = -120.0


def read_pcm(path: Path, start: float = 0.0, duration: float | None = None,
             rate: int = 16000) -> np.ndarray:
    """Mono float32 samples in -1..1 from any file ffmpeg can read.

    The whole stretch is held in memory, so callers keep requests to ten minutes or less.
    Asking past the end of the file gives fewer samples, or none."""
    cmd = [shell.ffmpeg(), "-v", "error", "-nostdin", "-ss", f"{max(0.0, start):.6f}"]
    if duration is not None:
        cmd += ["-t", f"{max(0.0, duration):.6f}"]
    cmd += ["-i", str(path), "-vn", "-sn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
    raw = shell.run_binary(cmd)
    return np.frombuffer(raw[:len(raw) - len(raw) % 4], dtype="<f4").astype(np.float32)


def level_db(pcm: np.ndarray) -> float:
    """RMS level in dBFS; -120.0 for silence or for no samples at all."""
    if len(pcm) == 0:
        return SILENCE_DB
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    return max(SILENCE_DB, 20.0 * float(np.log10(rms))) if rms > 0.0 else SILENCE_DB


def levels_db(pcm: np.ndarray, rate: int, hop: float) -> np.ndarray:
    """One RMS dBFS value per `hop` seconds: value i covers [i*hop, (i+1)*hop).

    A last part-filled hop is measured over the samples it has. Hop edges are placed from
    the clock, not by adding up a rounded hop length, so a hop that is not a whole number of
    samples (a video frame, say) does not drift."""
    count = int(np.ceil(len(pcm) / (rate * hop) - 1e-9)) if len(pcm) else 0
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    edges = np.minimum(np.round(np.arange(count + 1) * rate * hop).astype(np.int64), len(pcm))
    power = np.concatenate([[0.0], np.cumsum(np.square(pcm, dtype=np.float64))])
    sizes = np.maximum(np.diff(edges), 1)
    mean = np.maximum(np.diff(power[edges]) / sizes, 0.0)
    floor = 10.0 ** (SILENCE_DB / 10.0)
    return (10.0 * np.log10(np.maximum(mean, floor))).astype(np.float32)
