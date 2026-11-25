"""Where the streamer's camera sits in the frame, worked out once per stream.

Three outcomes. No steady small face: there is no camera box, and each clip is framed from
its own faces. A steady face with clean straight edges around it: that box. A steady face
with no clean edge (a cut-out streamer, a round frame): a safe box a few face-widths wide.

The face is looked for twice. First in the whole frame, which is cheap and finds a large
camera. When that shows no steady face, the frame is searched in overlapping parts, because a
small camera in a corner is invisible to the face model in the whole frame (`smallfaces`).

The edges are found in the average of many frames. A game changes from frame to frame and
averages to a blur, while the camera's border stays where it is.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import numpy as np

from clipper.models import Rect
from streamclip.framing.crops import FACE_DOWN, SAFE_FACE_WIDTHS, _largest, _place
from streamclip.schema import CameraBox

SAMPLE_SECONDS = 20.0           # how much of each window is looked at
FACE_FPS = 2.0                  # faces sampled per second
EDGE_FPS = 1.0                  # frames per second averaged for the edges
SEARCH_FPS = 0.5                # looks per second when the frame is searched in parts: each costs 26
SEARCH_WINDOWS = 12             # windows searched at most, spread across the stream

SAME_SPOT = 0.05                # face centres this close (fraction of the frame) are one spot
STEADY_SHARE = 0.60             # the spot holds a face in this share of the sampled moments
STEADY_WINDOWS = 0.50           # and shows up in this share of the windows
SMALL_FACE = 0.30               # a taller face than this is the whole picture, not a box

EDGE_WIDTH = 320                # frames are shrunk to this width before edges are measured
MIN_WIDTHS, MAX_WIDTHS = 1.5, 5.0       # a camera box is this many face-widths wide
MIN_SHAPE, MAX_SHAPE = 1.2, 1.9         # and this shape, width / height in pixels
STRONG = 2.0                    # a line this many times the picture's mean edge is worth trying
CLEAN = 1.5                     # every side must beat its surroundings by this many times the mean
WHOLE = 20                      # a side is as strong as its weakest stretch: this percentile
NEAR_BEST = 0.6                 # among boxes scoring this close to the best, the outermost wins
SMOOTH = 0.15                   # of a side's length, the stretch its strength is averaged over
TRIES = 16                      # lines tried on each side of the face (8 missed a box whose
                                # border was the ninth line out, on a tuning stream)

SAFE_SHAPE = 16 / 9
SAFE_CONFIDENCE = 0.3
DEFAULT_ASPECT = 16 / 9

_FRAME = Rect(0.0, 0.0, 1.0, 1.0)


def steady_face(detections_per_window: list[list[tuple[float, Rect, float]]]) -> Rect | None:
    """The median box of a face that sits in one small spot through most of the stream.

    Faces are grouped by where their centre is. The share is counted over sampled moments
    that show any face, so a face inside the game does not count against the streamer's.
    """
    seen = [(n, t, box) for n, window in enumerate(detections_per_window) for t, box, _ in window]
    if not seen:
        return None
    centres = np.array([(b.x + b.w / 2, b.y + b.h / 2) for _, _, b in seen])
    near = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2) <= SAME_SPOT
    busiest = centres[near[int(near.sum(axis=1).argmax())]]
    members = np.linalg.norm(centres - np.median(busiest, axis=0), axis=1) <= SAME_SPOT
    group = [s for s, inside in zip(seen, members) if inside]

    moments = {(n, t) for n, t, _ in seen}
    if len({(n, t) for n, t, _ in group}) < STEADY_SHARE * len(moments):
        return None
    if len({n for n, _, _ in group}) < STEADY_WINDOWS * len(detections_per_window):
        return None
    box = Rect(*(statistics.median(getattr(b, k) for _, _, b in group) for k in "xywh"))
    return box if box.h < SMALL_FACE else None


def _shrink(frame: np.ndarray) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    size = (EDGE_WIDTH, max(1, round(h * EDGE_WIDTH / w)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _edge_maps(frames: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Edge strength of the average frame: (upright lines, level lines), colours added up."""
    import cv2

    total, count = None, 0
    for frame in frames:
        small = _shrink(frame)
        if total is None:
            total = np.zeros_like(small)
        if small.shape != total.shape:
            continue                            # a window of another shape cannot be averaged in
        total += small
        count += 1
    if not count:
        return None
    mean = total / count
    if mean.ndim == 2:
        mean = mean[:, :, None]
    upright = sum(np.abs(cv2.Sobel(mean[:, :, c], cv2.CV_32F, 1, 0, ksize=3)) for c in range(mean.shape[2]))
    level = sum(np.abs(cv2.Sobel(mean[:, :, c], cv2.CV_32F, 0, 1, ksize=3)) for c in range(mean.shape[2]))
    return upright, level


def _lines(profile: np.ndarray, lo: float, hi: float, floor: float) -> list[tuple[int, float]]:
    """Strong peaks of `profile` between lo and hi, as (index, exact position in pixels)."""
    found = []
    for i in range(max(1, math.floor(lo)), min(len(profile) - 1, math.ceil(hi) + 1)):
        if profile[i] < floor or profile[i] <= profile[i - 1] or profile[i] < profile[i + 1]:
            continue
        around = profile[i - 1:i + 2]
        # A step between two pixels shows on both; the weighted middle is where it is.
        exact = float((around * np.arange(i - 1, i + 2)).sum() / around.sum()) + 0.5
        found.append((float(profile[i]), i, exact))
    found.sort(reverse=True)
    return [(i, exact) for _, i, exact in found[:TRIES]]


def _side(strength: np.ndarray, line: int, a: float, b: float) -> tuple[float, float] | None:
    """How one side of a box stands out: (its weakest stretch minus the same line carried on
    past its corners, its mean strength). `strength` is indexed [along, across]."""
    n = strength.shape[0]
    a, b = max(0, round(a)), min(n, round(b))
    if b - a < 4:
        return None
    across = strength[:, max(0, line - 1):line + 2].max(axis=1)      # a pixel of slack either way
    run = across[a + 1:b - 1]
    # Smoothed, so a short weak spot (the picture matching the game there) does not sink a
    # side, while a line that covers only part of it still does.
    span = max(3, round(SMOOTH * (b - a)))
    smooth = np.convolve(run, np.ones(span) / span, mode="valid") if len(run) >= span else run
    reach = max(3, round(0.2 * (b - a)))
    beyond = [across[max(0, a - 2 - reach):max(0, a - 2)], across[b + 2:b + 2 + reach]]
    carried_on = max((float(part.mean()) for part in beyond if len(part)), default=0.0)
    return float(np.percentile(smooth, WHOLE)) - carried_on, float(run.mean())


def box_from_edges(frames: Iterable[np.ndarray], face: Rect) -> tuple[Rect, float] | None:
    """The camera box around `face` from the straight edges that stay put, with a confidence.

    A side has to be strong along its whole length and stop at the corners, which rules out
    lines inside the camera picture and lines that run on across the game. A side on the
    frame's own edge has no line to find, so a box in a corner needs only its inner sides.
    """
    maps = _edge_maps(frames)
    if maps is None:
        return None
    upright, level = maps
    h, w = upright.shape
    normal = float(upright.mean() + level.mean()) / 2
    if normal <= 0:
        return None
    floor = STRONG * normal

    left, right = face.x * w, (face.x + face.w) * w
    top, bottom = face.y * h, (face.y + face.h) * h
    face_w = right - left
    reach_w, reach_h = MAX_WIDTHS * face_w, MAX_WIDTHS * face_w / MIN_SHAPE
    rows = slice(max(0, math.floor(top)), min(h, math.ceil(bottom) + 1))
    cols = slice(max(0, math.floor(left)), min(w, math.ceil(right) + 1))
    if rows.stop - rows.start < 1 or cols.stop - cols.start < 1:
        return None
    across, down = upright[rows].mean(axis=0), level[:, cols].mean(axis=1)

    # Each side: the lines found, plus the frame's own edge (None) when it is within reach,
    # for a box that sits flush against it.
    def tries(profile: np.ndarray, lo: float, hi: float, edge: float) -> list[tuple[int | None, float]]:
        return _lines(profile, lo, hi, floor) + ([(None, edge)] if lo <= edge <= hi else [])

    lefts = tries(across, right - reach_w, left, 0.0)
    rights = tries(across, right, left + reach_w, float(w))
    tops = tries(down, bottom - reach_h, top, 0.0)
    bottoms = tries(down, bottom, top + reach_h, float(h))

    scored = []
    for (li, lx), (ri, rx), (ti, ty), (bi, by) in product(lefts, rights, tops, bottoms):
        bw, bh = rx - lx, by - ty
        if bh <= 0 or not MIN_WIDTHS * face_w <= bw <= MAX_WIDTHS * face_w:
            continue
        if not MIN_SHAPE <= bw / bh <= MAX_SHAPE:
            continue
        if (li is None and ri is None) or (ti is None and bi is None):
            continue                            # nothing found across, or nothing found down
        sides = [_side(upright, i, ty, by) for i in (li, ri) if i is not None]
        sides += [_side(level.T, i, lx, rx) for i in (ti, bi) if i is not None]
        if any(s is None for s in sides):
            continue
        stands_out = min(s[0] for s in sides) / normal
        if stands_out >= CLEAN:
            strength = sum(s[1] for s in sides) / len(sides) / normal
            scored.append((len(sides), stands_out, bw * bh, strength,
                           Rect(lx / w, ty / h, bw / w, bh / h)))
    if not scored:
        return None
    # A box with every side found beats one that leans on the frame's edge.
    most = max(s[0] for s in scored)
    scored = [s for s in scored if s[0] == most]
    best = max(s[1] for s in scored)
    *_, strength, rect = max((s for s in scored if s[1] >= NEAR_BEST * best), key=lambda s: s[2])
    return rect, min(1.0, max(0.0, 1.0 - 1.0 / strength))


def safe_box(face: Rect, src_aspect: float) -> Rect:
    """A 16:9 box a few face-widths wide around the face, kept inside the frame."""
    shape = SAFE_SHAPE / src_aspect
    w, h = face.w * SAFE_FACE_WIDTHS, face.w * SAFE_FACE_WIDTHS / shape
    fit_w, fit_h = _largest(shape, _FRAME)
    if w > fit_w:
        w, h = fit_w, fit_h
    return _place(w, h, face.x + face.w / 2, face.y + face.h / 2, _FRAME, FACE_DOWN)


def _spread(items: list, most: int) -> list:
    """At most `most` of `items`, evenly spaced, the first and the last among them."""
    if len(items) <= most:
        return list(items)
    return [items[round(i * (len(items) - 1) / (most - 1))] for i in range(most)]


def find_camera(windows: list[Path], detect=None, search=None) -> CameraBox | None:
    """The camera box for the whole stream, from the candidates' low-resolution windows.

    `detect` looks at whole frames; `search` looks at them in parts and is used only when
    `detect` shows no steady face. Both default to the real face model. A caller that passes
    its own `detect` and no `search` gets no search.
    """
    from clipper.reframe import faces
    from streamclip.framing import smallfaces

    if detect is None:
        detect = smallfaces.detect_faces
        if search is None:
            def search(window, start, end, fps):
                return smallfaces.detect_faces(window, start, end, fps=fps, tiles=smallfaces.TILES)
    seen = [detect(window, 0.0, SAMPLE_SECONDS, fps=FACE_FPS) for window in windows]
    face = steady_face(seen)
    if face is None and search is not None:
        windows = _spread(windows, SEARCH_WINDOWS)
        seen = [search(window, 0.0, SAMPLE_SECONDS, fps=SEARCH_FPS) for window in windows]
        face = steady_face(seen)
    if face is None:
        return None

    # Only windows that show the steady face are averaged: an intermission screen has no box.
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    showing = [window for window, found in zip(windows, seen)
               if any(math.hypot(b.x + b.w / 2 - cx, b.y + b.h / 2 - cy) <= SAME_SPOT for _, b, _ in found)]
    aspect = []

    def frames():
        for window in showing:
            for _, frame in faces.iter_sampled_frames(window, 0.0, SAMPLE_SECONDS, fps=EDGE_FPS):
                if not aspect:
                    aspect.append(frame.shape[1] / frame.shape[0])
                yield _shrink(frame)

    found = box_from_edges(frames(), face)
    if found is not None:
        return CameraBox(rect=found[0], confidence=found[1])
    return CameraBox(rect=safe_box(face, aspect[0] if aspect else DEFAULT_ASPECT),
                     confidence=SAFE_CONFIDENCE)
