"""Faces that are small in the frame: a streamer's camera in a corner.

The face model is made for a face that fills a fair share of the picture it is shown. A corner
camera a quarter of the frame wide holds a face about a twentieth of the frame wide, and shown
the whole frame the model finds nothing: 0 looks of 40 on a real broadcast with the camera in
the bottom right corner, so no clip showed the streamer. Shown a third of the frame, the same
model found the face in 38 of 40.

So besides the whole frame, this looks at overlapping tiles (to search for a camera) or at
named regions (once the camera is known), and gives every face back as a fraction of the whole
frame. A search costs 26 looks a frame, so it is done on a few frames; a region costs one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from clipper.models import Rect

TILES = 3           # a search's tiles are a third of the frame each way and step by half a tile
GROW = 0.15         # a region is looked at this much wider on each side, as a share of its size
CUT = 0.02          # a face this near a view's cut edge may be half a face (a share of the view)
MIN_SIDE = 16       # pixels: a smaller view holds nothing the model could read

_WHOLE = Rect(0.0, 0.0, 1.0, 1.0)
_EDGE = 1e-6        # a view this near the frame's edge ends at it

Look = Callable[[np.ndarray], list[tuple[Rect, float]]]


def views(tiles: int = 0, regions: Iterable[Rect] = ()) -> list[Rect]:
    """What is looked at in every frame: the whole of it, each region a little wider, then the tiles."""
    out = [_WHOLE]
    for r in regions:
        x0, y0 = max(0.0, r.x - GROW * r.w), max(0.0, r.y - GROW * r.h)
        x1, y1 = min(1.0, r.x + r.w * (1 + GROW)), min(1.0, r.y + r.h * (1 + GROW))
        if x1 > x0 and y1 > y0:
            out.append(Rect(x0, y0, x1 - x0, y1 - y0))
    if tiles > 1:
        size = 1.0 / tiles
        starts = [i * size / 2 for i in range(2 * tiles - 1)]
        out += [Rect(x, y, size, size) for y in starts for x in starts]
    return out


def _cut_off(box: Rect, view: Rect) -> bool:
    """Whether a face sits against an edge where the view was cut out of the frame. Only part of
    it may be in the view; an overlapping view holds the whole of it."""
    return ((view.x > _EDGE and box.x < CUT)
            or (view.y > _EDGE and box.y < CUT)
            or (view.x + view.w < 1.0 - _EDGE and box.x + box.w > 1.0 - CUT)
            or (view.y + view.h < 1.0 - _EDGE and box.y + box.h > 1.0 - CUT))


def _same_face(a: Rect, b: Rect) -> bool:
    """Two answers are one face when either's centre is inside the other."""
    def holds(outer: Rect, inner: Rect) -> bool:
        cx, cy = inner.x + inner.w / 2, inner.y + inner.h / 2
        return outer.x <= cx <= outer.x + outer.w and outer.y <= cy <= outer.y + outer.h
    return holds(a, b) or holds(b, a)


def _in_frame(rgb: np.ndarray, wanted: list[Rect], look: Look) -> list[tuple[Rect, float]]:
    """Every face in one frame, each once, as fractions of the whole frame."""
    h, w = rgb.shape[:2]
    seen = []
    for view in wanted:
        x0, y0 = round(view.x * w), round(view.y * h)
        x1, y1 = round((view.x + view.w) * w), round((view.y + view.h) * h)
        if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
            continue
        part = rgb if (x0, y0, x1, y1) == (0, 0, w, h) else np.ascontiguousarray(rgb[y0:y1, x0:x1])
        for box, mouth in look(part):
            if _cut_off(box, view):
                continue
            vw, vh = (x1 - x0) / w, (y1 - y0) / h
            seen.append((view.w * view.h,
                         Rect(x0 / w + box.x * vw, y0 / h + box.y * vh, box.w * vw, box.h * vh), mouth))
    seen.sort(key=lambda s: s[0])           # the closest look at a face measures it best
    kept: list[tuple[Rect, float]] = []
    for _, box, mouth in seen:
        if not any(_same_face(box, other) for other, _ in kept):
            kept.append((box, mouth))
    return kept


def _model_file() -> Path:
    from clipper import shell
    from clipper.reframe import faces

    path = shell.project_home() / "assets" / faces.MODEL_FILE
    if not path.exists():
        from clipper import provision
        provision.ensure(("model",))            # downloads it, once
    if not path.exists():
        raise FileNotFoundError(f"The face model is missing: {path}. Run `streamclip warm` to download it.")
    return path


def _model_look() -> tuple[Look, Callable[[], None]]:
    """The real face model as a `look`, and what closes it. Each picture is read on its own
    (no tracking from one to the next), because the views of one frame are different pictures."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    from clipper.reframe import faces

    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_model_file())),
        num_faces=4, running_mode=vision.RunningMode.IMAGE))

    def look(rgb: np.ndarray) -> list[tuple[Rect, float]]:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for marks in result.face_landmarks:
            xs, ys = [m.x for m in marks], [m.y for m in marks]
            box = Rect(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))
            mouth = abs(marks[faces.LOWER_LIP].y - marks[faces.UPPER_LIP].y) / box.h if box.h else 0.0
            out.append((box, mouth))
        return out

    return look, landmarker.close


def detect_faces(video: Path, start: float, end: float, fps: float = 10.0, tiles: int = 0,
                 regions: Iterable[Rect] = (), look: Look | None = None) -> list[tuple[float, Rect, float]]:
    """Every face in [start, end) as (seconds, box, mouth openness), small ones included.

    The whole frame is always looked at. `tiles=TILES` also searches it in overlapping parts;
    `regions` also looks inside each named part (a known camera box). `look` stands in for the
    face model in tests: it is given one picture and answers with the faces in it.
    """
    from clipper.reframe import faces

    wanted = views(tiles, list(regions))
    close = None
    if look is None:
        look, close = _model_look()
    found: list[tuple[float, Rect, float]] = []
    try:
        for t, rgb in faces.iter_sampled_frames(video, start, end, fps):
            found += [(t, box, mouth) for box, mouth in _in_frame(rgb, wanted, look)]
    finally:
        if close is not None:
            close()
    return found
