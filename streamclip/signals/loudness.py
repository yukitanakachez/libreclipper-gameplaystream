"""How loud each second is against the stream's own normal.

Everything here works on one RMS level per second. A loud streamer is not "exciting" all
the time, so loudness is measured against the surrounding ten minutes, and a surge is a
jump from quiet to louder than normal within a few seconds.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from streamclip.audio import SILENCE_DB, levels_db, read_pcm

# Starting points. Lane N tunes them on real streams.
BASELINE_WINDOW = 600       # seconds around each second that make up "normal"
BASELINE_FLOOR_DB = -60.0   # normal is never taken to be quieter than this
LOUD_FULL_DB = 12.0         # this far over normal, or more, is fully loud
SURGE_LOOKBACK = 3          # seconds before each second that a surge is measured from
SURGE_MIN_DB = 6.0          # a rise of this much is where a surge starts to count
SURGE_FULL_DB = 18.0        # a rise of this much, or more, is a full surge
MUTED_DB = -70.0            # below this is digital silence
MUTED_AFTER = 30            # a silence counts as muted from this second of it on

RATE = 16000
CHUNK = 600                 # seconds of sound held in memory at once


def seconds(duration: float) -> int:
    """How many per-second values a stream of this length has. A part second counts."""
    return max(0, math.ceil(duration - 1e-3))


def per_second_db(audio: Path, duration: float) -> np.ndarray:
    """One RMS level in dBFS per second of the stream.

    The sound is read `CHUNK` seconds at a time, so a 6-hour stream never sits in memory at
    once. Where the file ends before `duration`, the rest is silence."""
    count = seconds(duration)
    out = np.full(count, SILENCE_DB, dtype=np.float32)
    for start in range(0, count, CHUNK):
        pcm = read_pcm(audio, start=float(start), duration=float(min(CHUNK, count - start)),
                       rate=RATE)
        if len(pcm) == 0:
            break
        got = levels_db(pcm, RATE, 1.0)[: count - start]
        out[start:start + len(got)] = got
    return out


def moving_sum(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """`values` under a sliding odd-length kernel, centred, the same length as `values`
    however short it is. Past the ends counts as zero."""
    if len(values) == 0:
        return np.zeros(0)
    half = len(kernel) // 2
    return np.convolve(values, kernel)[half:half + len(values)]


def rolling_median(values: np.ndarray, before: int, after: int) -> np.ndarray:
    """The median of values[i-before : i+after+1] for every i, with the window cut off at
    the ends of the series, not padded. NaN where the window holds nothing."""
    values = np.asarray(values, dtype=np.float64)
    n, width = len(values), before + after + 1
    out = np.full(n, np.nan)
    if n == 0:
        return out
    padded = np.concatenate([np.full(before, np.nan), values, np.full(after, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    rows = max(1, 4_000_000 // width)               # about 32 MB of windows at a time
    for at in range(0, n, rows):
        block = windows[at:at + rows]
        filled = ~np.isnan(block).all(axis=1)
        out[at:at + rows][filled] = np.nanmedian(block[filled], axis=1)
    return out


def baseline(db: np.ndarray) -> np.ndarray:
    """What "normal" is at each second: the median level of the surrounding ten minutes."""
    half = BASELINE_WINDOW // 2
    return np.maximum(rolling_median(db, half, half), BASELINE_FLOOR_DB)


def over_baseline(db: np.ndarray) -> np.ndarray:
    """Each second's level minus normal, in dB."""
    db = np.asarray(db, dtype=np.float64)
    return db - baseline(db)


def relative_loudness(db: np.ndarray) -> np.ndarray:
    """0..1 per second: 0 at normal or below, 1 at `LOUD_FULL_DB` over normal or more."""
    return np.clip(over_baseline(db) / LOUD_FULL_DB, 0.0, 1.0)


def rise(db: np.ndarray) -> np.ndarray:
    """How many dB each second stands over the quietest of the `SURGE_LOOKBACK` seconds
    before it.

    The quiet it rises from is never taken to be more than `SURGE_MIN_DB` under normal, so
    sound coming back to normal after a silence is not a surge. Only ending up louder than
    normal is."""
    db = np.asarray(db, dtype=np.float64)
    out = np.zeros(len(db))
    if len(db) < 2:
        return out
    padded = np.concatenate([np.full(SURGE_LOOKBACK, np.inf), db[:-1]])
    lowest = np.lib.stride_tricks.sliding_window_view(padded, SURGE_LOOKBACK).min(axis=1)
    lowest[0] = db[0]                               # the first second has nothing before it
    return np.maximum(db - np.maximum(lowest, baseline(db) - SURGE_MIN_DB), 0.0)


def surge(db: np.ndarray) -> np.ndarray:
    """0..1 per second: a jump from quiet to loud. `SURGE_MIN_DB` is 0, `SURGE_FULL_DB` is 1."""
    return np.clip((rise(db) - SURGE_MIN_DB) / (SURGE_FULL_DB - SURGE_MIN_DB), 0.0, 1.0)


def muted(db: np.ndarray) -> np.ndarray:
    """True per second from the `MUTED_AFTER`th second of any run below `MUTED_DB`."""
    silent = np.asarray(db) < MUTED_DB
    out = np.zeros(len(silent), dtype=bool)
    run = 0
    for i, quiet in enumerate(silent):
        run = run + 1 if quiet else 0
        out[i] = run >= MUTED_AFTER
    return out
