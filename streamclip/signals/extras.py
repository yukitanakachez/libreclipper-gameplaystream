"""The two extra signals: chat bursts and the clips viewers already made.

Both are optional. With no chat or no clips the function gives None and the curve shares
that signal's weight among the rest.
"""

from __future__ import annotations

import math

import numpy as np

from streamclip.schema import ViewerClip
from streamclip.signals.loudness import moving_sum, rolling_median, seconds

# Starting points. Lane N tunes them on real streams.
CHAT_SMOOTH = 5             # seconds the message rate is averaged over
CHAT_BASELINE = 600         # seconds before each second that make up the normal rate
CHAT_WARMUP = 60            # the stream's first seconds are measured against themselves
CHAT_MIN_RATE = 0.5         # messages a second; normal is never taken to be slower
CHAT_LAG = 6                # seconds chat reacts late, so the burst is moved this much earlier
CHAT_MIN_RATIO = 2.0        # this many times the normal rate is where a burst starts to count
CHAT_FULL_RATIO = 5.0       # this many times, or more, is a full burst


def chat_ratio(times: list[float] | None, duration: float) -> np.ndarray | None:
    """Per second: the message rate as a multiple of the recent normal, already moved
    `CHAT_LAG` seconds earlier. None when there is no chat."""
    count = seconds(duration)
    if not times or count == 0:
        return None
    stamps = np.asarray(times, dtype=np.float64)
    stamps = stamps[np.isfinite(stamps) & (stamps >= 0) & (stamps < count)]
    per_second = np.bincount(stamps.astype(np.int64), minlength=count).astype(np.float64)
    box = np.ones(CHAT_SMOOTH)
    rate = moving_sum(per_second, box) / moving_sum(np.ones(count), box)
    normal = np.empty(count)
    normal[1:] = rolling_median(rate, CHAT_BASELINE - 1, 0)[:-1]      # the seconds before, not this one
    normal[:CHAT_WARMUP] = np.median(rate[:CHAT_WARMUP])
    ratio = rate / np.maximum(normal, CHAT_MIN_RATE)
    return np.concatenate([ratio[CHAT_LAG:], np.zeros(min(CHAT_LAG, count))])


def chat_burst(times: list[float] | None, duration: float) -> np.ndarray | None:
    """0..1 per second: `CHAT_MIN_RATIO` times the normal rate is 0, `CHAT_FULL_RATIO` is 1."""
    ratio = chat_ratio(times, duration)
    if ratio is None:
        return None
    return np.clip((ratio - CHAT_MIN_RATIO) / (CHAT_FULL_RATIO - CHAT_MIN_RATIO), 0.0, 1.0)


def _spans(clips: list[ViewerClip], count: int) -> list[tuple[int, int, int]]:
    """(first second, one past the last second, views) of each clip that lies in the stream."""
    spans = []
    for clip in clips:
        first, last = max(0, math.floor(clip.offset)), min(count, math.ceil(clip.offset + clip.duration))
        if first < last:
            spans.append((first, last, max(0, int(clip.views))))
    return spans


def clip_bump(clips: list[ViewerClip] | None, duration: float) -> np.ndarray | None:
    """0..1 per second: each clip adds log10(views + 1) / log10(most views + 1) across its
    own span, capped at 1. None when there are no clips."""
    if not clips:
        return None
    count = seconds(duration)
    bump = np.zeros(count)
    spans = _spans(clips, count)
    most = math.log10(max((views for _, _, views in spans), default=0) + 1)
    for first, last, views in spans:
        bump[first:last] += math.log10(views + 1) / most if most else 1.0
    return np.minimum(bump, 1.0)


def clip_views(clips: list[ViewerClip] | None, duration: float) -> np.ndarray | None:
    """Per second: the views of the most-watched clip covering it, 0 where there is none."""
    if not clips:
        return None
    count = seconds(duration)
    views = np.zeros(count, dtype=np.int64)
    for first, last, seen in _spans(clips, count):
        views[first:last] = np.maximum(views[first:last], seen)
    return views
