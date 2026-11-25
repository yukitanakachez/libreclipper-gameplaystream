"""The excitement curve: every signal weighed into one value per second.

Sound is the backbone. Chat and viewer clips are extras: when one is missing its weight is
shared among the rest, so a stream without chat is scored on the same 0..1 scale as one
with it. Music with nobody talking, and muted sound, are marked down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from streamclip.schema import Signals, ViewerClip
from streamclip.signals import events, extras, loudness

# Tuned on six real streams (docs/stream-tuning-notes.md). The voice weight looks large
# because the classifier's scores are small: its strongest second in a thousand scores about
# 0.3, where loudness, chat and viewer clips all reach 1.
WEIGHTS = {"loudness": 0.17, "surge": 0.08, "voice": 0.39, "chat": 0.22, "clips": 0.14}
MUSIC_MIN = 0.6             # music scored over this ...
SPEECH_MAX = 0.2            # ... with speech under this ...
MUSIC_RUN = 20              # ... for this many seconds or more is an intermission


@dataclass
class Details:
    """What lay behind each second's scores, kept so a hot spot can say why it was picked.

    `build_signals` hangs this on the Signals it returns, as `signals.details`. It is not
    part of signals.json. Signals read back from a file have none, and `pick_hotspots` then
    words its reasons from the 0..1 scores alone."""
    label: list[str]                    # the voice event's name, or ""
    over_db: np.ndarray                 # dB over the surrounding ten minutes
    rise_db: np.ndarray                 # dB over the quietest of the seconds just before
    chat_ratio: np.ndarray | None       # message rate as a multiple of normal
    clip_views: np.ndarray | None       # views of the most-watched clip covering the second


def penalty(muted: np.ndarray, music: np.ndarray, speech: np.ndarray) -> np.ndarray:
    """0..1 per second: 1 where the sound is muted, and the music score where music plays
    with nobody talking for `MUSIC_RUN` seconds or more."""
    out = np.zeros(len(muted))
    alone = (np.asarray(music) > MUSIC_MIN) & (np.asarray(speech) < SPEECH_MAX)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], alone, [False]]).astype(np.int8)))
    for first, last in zip(edges[::2], edges[1::2]):
        if last - first >= MUSIC_RUN:
            out[first:last] = np.asarray(music)[first:last]
    out[np.asarray(muted, dtype=bool)] = 1.0
    return np.clip(out, 0.0, 1.0)


def combine(parts: dict[str, np.ndarray | None], marked_down: np.ndarray) -> np.ndarray:
    """The weighted sum of the signals that exist, times (1 - penalty), kept to 0..1.

    A signal that is None gives its weight to the others in proportion to theirs."""
    present = {name: np.asarray(values, dtype=np.float64)
               for name, values in parts.items() if values is not None}
    total = sum(WEIGHTS[name] for name in present)
    mixed = sum(WEIGHTS[name] / total * values for name, values in present.items())
    return np.clip(mixed * (1.0 - np.asarray(marked_down)), 0.0, 1.0)


def _plain(values: np.ndarray | None) -> list[float] | None:
    return None if values is None else [round(float(v), 4) for v in values]


def build_signals(audio: Path, duration: float, chat_times: list[float] | None,
                  viewer_clips: list[ViewerClip] | None, classify=None) -> Signals:
    """Every signal for the whole stream, and the excitement curve made from them.

    `classify` is the sound classifier (see `events.voice_events`); None means the real one."""
    db = loudness.per_second_db(audio, duration)
    track = events.voice_events(audio, duration, classify)
    parts = {
        "loudness": loudness.relative_loudness(db),
        "surge": loudness.surge(db),
        "voice": track.voice,
        "chat": extras.chat_burst(chat_times, duration),
        "clips": extras.clip_bump(viewer_clips, duration),
    }
    marked_down = penalty(loudness.muted(db), track.music, track.speech)
    signals = Signals(duration=float(duration), penalty=_plain(marked_down),
                      excitement=_plain(combine(parts, marked_down)),
                      **{name: _plain(values) for name, values in parts.items()})
    signals.details = Details(label=track.label, over_db=loudness.over_baseline(db),
                              rise_db=loudness.rise(db),
                              chat_ratio=extras.chat_ratio(chat_times, duration),
                              clip_views=extras.clip_views(viewer_clips, duration))
    return signals
