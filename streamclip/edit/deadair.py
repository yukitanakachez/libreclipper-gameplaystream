"""Dead air: stretches where nobody talks and nothing happens.

A stretch is dead when there is no speech, the sound is quiet against the clip's own level,
and the picture is nearly still. A dead stretch longer than the threshold is cut down to a
short breath. A quiet stretch where the screen is the joke moves, so it stays; a pause the
judge protected stays whatever it looks like.

Intervals are in source-clip time (0 is the clip's first frame), like `clipper.pacing`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from clipper.models import Word
from streamclip import audio, settings

STEP = 0.05                 # seconds; the clip is judged in steps this long
SPEECH_PAD = 0.05           # seconds either side of a word that still count as speech

# Starting points, to be tuned on real streams (Lane N).
QUIET_BELOW_DB = 20.0       # quiet: this far under the clip's reference level
STILL_BELOW = 1.5           # still: mean frame difference under this, on a 0..255 scale
SPEECH_FLOOR_DB = -60.0     # words measured under this carry no sound, so they set no reference


def motion_series(video: Path, fps: float = 10.0) -> np.ndarray:
    """How much the picture changes, `fps` times a second: value i is the mean absolute
    difference (0..255) between sample i and the one before it. The first value is 0."""
    import cv2

    from clipper import shell
    from clipper.reframe.faces import iter_sampled_frames

    duration = float(shell.probe(video)["format"]["duration"])
    values: list[float] = []
    before = None
    for _, frame in iter_sampled_frames(Path(video), 0.0, duration, fps=fps):
        small = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_RGB2GRAY).astype(np.float32)
        values.append(0.0 if before is None else float(np.abs(small - before).mean()))
        before = small
    return np.asarray(values, dtype=np.float32)


def _steps(a: float, b: float, count: int) -> slice:
    """The steps a span [a, b) of source-clip seconds touches."""
    first = max(0, math.floor(a / STEP + 1e-6))
    last = min(count, math.ceil(b / STEP - 1e-6))
    return slice(first, max(first, last))


def _reference_db(levels: np.ndarray, speech: np.ndarray) -> float:
    """The level "quiet" is measured against: the median over the words, or with no words
    (or words that fell on silence) the clip's 90th-percentile level."""
    if len(levels) == 0:
        return audio.SILENCE_DB
    spoken = levels[speech[:len(levels)]]
    if len(spoken):
        median = float(np.median(spoken))
        if median > SPEECH_FLOOR_DB:
            return median
    return float(np.percentile(levels, 90))


def _dead_steps(words: list[Word], clip_start: float, clip_end: float, pcm: np.ndarray,
                rate: int, motion: np.ndarray, motion_fps: float,
                protect: list[tuple[float, float]]) -> np.ndarray:
    length = clip_end - clip_start
    count = max(0, math.ceil(length / STEP - 1e-6))

    speech = np.zeros(count, dtype=bool)
    for w in words:
        if w.end > clip_start and w.start < clip_end:
            speech[_steps(w.start - clip_start - SPEECH_PAD, w.end - clip_start + SPEECH_PAD, count)] = True

    # Sound or picture that was not measured is never called quiet or still.
    levels = audio.levels_db(np.asarray(pcm, dtype=np.float32), rate, STEP)[:count]
    quiet = np.zeros(count, dtype=bool)
    quiet[:len(levels)] = levels < _reference_db(levels, speech) - QUIET_BELOW_DB

    # motion[i] is the change over ((i-1)/fps, i/fps], so a step asks the sample that ends it.
    motion = np.asarray(motion, dtype=np.float32)
    middles = (np.arange(count) + 0.5) * STEP
    index = np.floor(middles * motion_fps + 1e-6).astype(np.int64) + 1
    still = np.zeros(count, dtype=bool)
    known = index < len(motion)
    still[known] = motion[index[known]] < STILL_BELOW

    dead = ~speech & quiet & still
    for a, b in protect:
        dead[_steps(a - clip_start, b - clip_start, count)] = False
    return dead


def _removals(dead: np.ndarray, length: float, dead_air: float) -> list[tuple[float, float]]:
    """Runs of dead steps longer than `dead_air`, less half a breath at each end."""
    half = settings.BREATH / 2
    removals: list[tuple[float, float]] = []
    padded = np.concatenate([[False], dead, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    for first, last in zip(edges[::2], edges[1::2]):
        start, end = first * STEP, min(last * STEP, length)
        if end - start > dead_air + 1e-6 and end - start > settings.BREATH:
            removals.append((start + half, end - half))
    return removals


def keep_intervals(words: list[Word], clip_start: float, clip_end: float, pcm: np.ndarray,
                   rate: int, motion: np.ndarray, motion_fps: float,
                   protect: list[tuple[float, float]], fps: float,
                   dead_air: float = settings.DEAD_AIR) -> list[tuple[float, float]]:
    """Source-clip [a, b) spans to keep, in order, with their edges on the frame grid.

    `words` and `protect` are in stream seconds; `pcm` is the clip's own sound and `motion`
    its own `motion_series`, both starting at `clip_start`."""
    length = clip_end - clip_start
    dead = _dead_steps(words, clip_start, clip_end, pcm, rate, motion, motion_fps, protect)

    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for cut_start, cut_end in _removals(dead, length, dead_air):
        if cut_start > cursor:
            kept.append((cursor, cut_start))
        cursor = cut_end
    if cursor < length:
        kept.append((cursor, length))

    # As clipper.pacing.keep_intervals: edges only ever move outward, so kept spans grow.
    aligned: list[tuple[float, float]] = []
    for a, b in kept:
        a2 = math.floor(a * fps + 1e-9) / fps
        b2 = min(math.ceil(b * fps - 1e-9) / fps, math.floor(length * fps + 1e-9) / fps)
        if b2 > a2:
            if aligned and a2 <= aligned[-1][1]:
                aligned[-1] = (aligned[-1][0], b2)
            else:
                aligned.append((a2, b2))
    return aligned
