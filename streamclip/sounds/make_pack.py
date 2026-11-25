"""Synthesise the sound pack from nothing but numpy and scipy, so every sound is ours.

Run `python -m streamclip.sounds.make_pack` to rewrite the four files and `manifest.json` in
`streamclip/assets/sounds`. Every random number comes from one seeded generator, so the files
come out byte for byte the same each time. Change a recipe here, run it, and commit both.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
from scipy import signal

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "sounds"
RATE = 48000
PEAK_DB = -1.05                     # just under -1 dBFS, so rounding to 16 bits never crosses it
SOURCE = "synthesised by streamclip/sounds/make_pack.py"
LICENCE = "CC0-1.0"


def _time(seconds: float) -> np.ndarray:
    return np.arange(round(seconds * RATE)) / RATE


def _sweep(t: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """A sine whose frequency glides from f0 to f1 on a log scale across t."""
    freq = f0 * (f1 / f0) ** (t / t[-1])
    return np.sin(2 * np.pi * np.cumsum(freq) / RATE)


def boom(rng: np.random.Generator) -> np.ndarray:
    """The payoff: a low falling thud with a click on the front and a short tail."""
    t = _time(0.9)
    body = _sweep(t, 90, 38) * np.exp(-t / 0.28)
    click = np.zeros_like(t)
    n = round(0.015 * RATE)
    sos = signal.butter(2, 1500, "lowpass", fs=RATE, output="sos")
    click[:n] = signal.sosfilt(sos, rng.standard_normal(n)) * np.exp(-np.arange(n) / (0.004 * RATE))
    dry = np.tanh(1.8 * (body + 0.5 * click)) / np.tanh(1.8)
    dry[-round(0.02 * RATE):] *= np.linspace(1.0, 0.0, round(0.02 * RATE))
    out = np.zeros(len(dry) + round(0.18 * RATE))
    out[:len(dry)] += dry
    for delay, gain in ((0.06, 0.22), (0.12, 0.12), (0.18, 0.06)):
        start = round(delay * RATE)
        out[start:start + len(dry)] += gain * dry
    return out


def whoosh(rng: np.random.Generator) -> np.ndarray:
    """The hard cut: noise through a band that sweeps upward, swelling and fading."""
    t = _time(0.45)
    noise = rng.standard_normal(len(t))
    nperseg = 512
    freqs, frames, spec = signal.stft(noise, fs=RATE, nperseg=nperseg)
    centre = 300 * (3000 / 300) ** (frames / frames[-1])
    # A band-pass one octave wide on a log scale, its centre moving frame by frame.
    distance = np.log2(np.maximum(freqs[:, None], 1.0) / centre[None, :])
    spec *= np.exp(-0.5 * (distance / 0.5) ** 2)
    _, out = signal.istft(spec, fs=RATE, nperseg=nperseg)
    out = out[:len(t)]
    bell = np.exp(-0.5 * ((t - 0.5 * t[-1]) / (0.2 * t[-1])) ** 2)
    return out * bell


def pop(rng: np.random.Generator) -> np.ndarray:
    """The reaction: a short rising blip."""
    t = _time(0.12)
    attack = np.minimum(t / 0.003, 1.0)
    return (_sweep(t, 880, 1320) + 0.25 * _sweep(t, 1760, 2640)) * attack * np.exp(-t / 0.035)


# id, recipe, role (a Beat kind), gain_db, lead in seconds. There is deliberately no rising
# sound before a moment (a "riser"): Yuki had it removed.
RECIPES = [
    ("boom", boom, "payoff", -8.0, 0.0),
    ("whoosh", whoosh, "hard_cut", -12.0, 0.1),
    ("pop", pop, "reaction", -12.0, 0.0),
]


def _write(path: Path, x: np.ndarray) -> None:
    x = x * (10 ** (PEAK_DB / 20) / np.abs(x).max())
    pcm = np.round(x * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def main(dest: Path = ASSETS) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for sound_id, recipe, role, gain_db, lead in RECIPES:
        x = recipe(rng)
        name = f"{sound_id}.wav"
        _write(dest / name, x)
        seconds = round(len(x) / RATE, 4)
        rows.append({"id": sound_id, "file": name, "role": role, "gain_db": gain_db,
                     "lead": lead, "seconds": seconds,
                     "source": SOURCE, "licence": LICENCE})
    (dest / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
