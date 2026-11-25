"""Where the sounds and the zoom go.

The judge names the beats (payoff, hard cut, reaction) in stream seconds. Sounds
are placed in cut time, a few at most and well apart; the zoom goes to the loudest reaction,
in source-clip time, because the layout's shots are on that clock.
"""

from __future__ import annotations

import numpy as np

from clipper.pacing import TimeMap
from streamclip import audio, settings
from streamclip.schema import Beat, Placed, Sound

ROLE_ORDER = ("payoff", "hard_cut", "reaction")     # who gets a sound first; a build-up never does
ZOOM_LISTEN = 0.5           # seconds of sound measured around each reaction


def place_sounds(beats: list[Beat], clip_start: float, time_map: TimeMap, speed: float,
                 pack: dict[str, Sound]) -> list[Placed]:
    """One sound per beat, in cut time, sorted by when it starts.

    A sound with a lead starts that much before its beat. Sounds stay `SOUND_GAP` finished
    seconds apart and there are never more than `MAX_SOUNDS`; when two beats compete, the
    payoff wins, then the hard cut, then the reaction."""
    if not time_map.intervals:
        return []
    clip_length = time_map.intervals[-1][1]
    gap = settings.SOUND_GAP * speed
    wanted = sorted((b for b in beats if b.kind in ROLE_ORDER and b.kind in pack),
                    key=lambda b: (ROLE_ORDER.index(b.kind), b.t))
    placed: list[Placed] = []
    for beat in wanted:
        if len(placed) >= settings.MAX_SOUNDS:
            break
        t = beat.t - clip_start
        if not 0.0 <= t <= clip_length:
            continue
        sound = pack[beat.kind]
        at = max(0.0, time_map(t) - sound.lead)
        if all(abs(at - p.at) >= gap - 1e-9 for p in placed):
            placed.append(Placed(sound.id, sound.path, at, sound.gain_db))
    return sorted(placed, key=lambda p: p.at)


def zoom_at(beats: list[Beat], clip_start: float, pcm: np.ndarray, rate: int) -> float | None:
    """When the face zoom goes, in source-clip time: the loudest reaction, else the first
    payoff, else never. `pcm` is the clip's own sound."""
    length = len(pcm) / rate
    inside = sorted((b.t - clip_start, b.kind) for b in beats if 0.0 <= b.t - clip_start <= length)

    best, best_db = None, None
    for t, kind in inside:
        if kind != "reaction":
            continue
        a = max(0, int((t - ZOOM_LISTEN / 2) * rate))
        db = audio.level_db(pcm[a:int((t + ZOOM_LISTEN / 2) * rate)])
        if best_db is None or db > best_db:
            best, best_db = t, db
    if best is not None:
        return best
    return next((t for t, kind in inside if kind == "payoff"), None)
