"""The shortlist: the highest peaks of the excitement curve, each with why it was picked."""

from __future__ import annotations

import math

import numpy as np

from streamclip.schema import Fired, Hotspot, Signals
from streamclip.settings import PEAK_GAP, PEAKS_PER_CLIP, WINDOW_AFTER, WINDOW_BEFORE
from streamclip.signals import extras, loudness

SMOOTH = 5                  # seconds the curve is smoothed over before peaks are taken
SPREAD_FROM = 90 * 60       # streams this long or longer are spread over their length:
SPREAD_BLOCK = 30 * 60      # no block of this many seconds ...
SPREAD_SHARE = 3            # ... holds more than one in this many of the shortlist
FIRED_OVER = 0.5            # a signal over this ...
FIRED_WITHIN = 5            # ... within this many seconds of the peak is a reason

# The part signals in the order they are listed when equally strong: (Signals field, Fired name).
_PARTS = (("voice", "voice"), ("loudness", "loudness"), ("surge", "surge"), ("chat", "chat"),
          ("clips", "viewer_clip"))


def smooth(excitement: np.ndarray) -> np.ndarray:
    """The curve averaged over `SMOOTH` seconds, the middle second counting most, so a
    one-second spike stays a peak at its own second."""
    half = SMOOTH // 2 + 1
    kernel = np.concatenate([np.arange(1, half + 1), np.arange(half - 1, 0, -1)]).astype(np.float64)
    return loudness.moving_sum(excitement, kernel / kernel.sum())


def _number(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}".rstrip("0").rstrip(".")


def _views(views: int) -> str:
    if views >= 1_000_000:
        return f"{_number(views / 1_000_000)}M views"
    if views >= 1000:
        return f"{_number(views / 1000)}k views"
    return f"{views} view" if views == 1 else f"{views} views"


def _detail(signal: str, at: int, strength: float, details) -> str:
    """Why this signal fired, in words the judge and the test page can show as they are."""
    more = " or more" if strength >= 1.0 else ""
    if signal == "voice":
        label = details.label[at] if details is not None else ""
        return label or "shouting, laughter or cheering"
    if signal == "loudness":
        over, tail = ((details.over_db[at], "") if details is not None
                      else (strength * loudness.LOUD_FULL_DB, more))
        return f"+{over:.0f} dB{tail} over the ten minutes around it"
    if signal == "surge":
        span = loudness.SURGE_FULL_DB - loudness.SURGE_MIN_DB
        rise, tail = ((details.rise_db[at], "") if details is not None
                      else (loudness.SURGE_MIN_DB + strength * span, more))
        return f"jumped {rise:.0f} dB{tail} within {loudness.SURGE_LOOKBACK} seconds"
    if signal == "chat":
        span = extras.CHAT_FULL_RATIO - extras.CHAT_MIN_RATIO
        ratio, tail = ((details.chat_ratio[at], "")
                       if details is not None and details.chat_ratio is not None
                       else (extras.CHAT_MIN_RATIO + strength * span, more))
        return f"chat ×{_number(ratio)}{tail}"
    if details is not None and details.clip_views is not None:
        return f"viewer clip, {_views(int(details.clip_views[at]))}"
    return "viewer clip"


def _fired(signals: Signals, peak: int) -> list[Fired]:
    """Every part signal over `FIRED_OVER` within `FIRED_WITHIN` seconds of the peak,
    strongest first. Each is reported at its own strongest second."""
    details = getattr(signals, "details", None)
    first = max(0, peak - FIRED_WITHIN)
    found = []
    for field, name in _PARTS:
        values = getattr(signals, field)
        if values is None:
            continue
        near = np.asarray(values[first:peak + FIRED_WITHIN + 1], dtype=np.float64)
        if len(near) == 0 or near.max() <= FIRED_OVER:
            continue
        at = first + int(np.argmax(near))
        strength = float(near.max())
        found.append(Fired(signal=name, t=float(at), strength=round(strength, 3),
                           detail=_detail(name, at, strength, details)))
    return sorted(found, key=lambda f: -f.strength)


def pick_hotspots(signals: Signals, wanted: int) -> list[Hotspot]:
    """Up to `PEAKS_PER_CLIP × wanted` hot spots, numbered from 1 in time order.

    The highest second of the smoothed curve is taken, everything within `PEAK_GAP` of it
    is ruled out, and so on until there are enough or nothing above zero is left. A short or
    flat stream gives fewer. Each hot spot's window runs `WINDOW_BEFORE` seconds before the
    peak to `WINDOW_AFTER` after, cut off at the ends of the stream."""
    left = smooth(np.asarray(signals.excitement, dtype=np.float64))
    target = PEAKS_PER_CLIP * max(0, int(wanted))
    spread = signals.duration >= SPREAD_FROM
    most_per_block = math.ceil(target / SPREAD_SHARE)
    held: dict[int, int] = {}
    gap = math.ceil(PEAK_GAP) - 1                   # seconds each side that a peak rules out
    peaks: list[tuple[int, float]] = []
    while len(peaks) < target and len(left):
        at = int(np.argmax(left))
        if left[at] <= 0.0:
            break
        block = at // SPREAD_BLOCK
        if spread and held.get(block, 0) >= most_per_block:
            left[block * SPREAD_BLOCK:(block + 1) * SPREAD_BLOCK] = 0.0
            continue
        peaks.append((at, float(left[at])))
        held[block] = held.get(block, 0) + 1
        left[max(0, at - gap):at + gap + 1] = 0.0
    return [Hotspot(n=n, peak=float(at), start=max(0.0, at - WINDOW_BEFORE),
                    end=min(float(signals.duration), at + WINDOW_AFTER),
                    score=round(score, 4), fired=_fired(signals, at))
            for n, (at, score) in enumerate(sorted(peaks), start=1)]
