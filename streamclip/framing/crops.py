"""Crop rectangles for the three layouts, as fractions of the source frame.

A crop worked out on the 360p window applies unchanged to the 1080p stretch. `src_aspect` is
the source's width divided by its height, which is all a fraction needs to know about pixels.
"""

from __future__ import annotations

from clipper.models import Rect
from clipper.reframe.layout import single_crop
from streamclip.settings import FACE_H, GAME_H, OUT_H, OUT_W, SAFE_TOP

FACE_ASPECT = OUT_W / FACE_H        # the face panel, width / height in pixels
GAME_ASPECT = OUT_W / GAME_H        # the game panel
SAFE_FACE_WIDTHS = 3.2              # with no camera box, the face panel is this many faces wide
FACE_DOWN = 0.45                    # the face's centre this far down camera.py's stand-in box
# The top SAFE_TOP of the finished frame is under the app's tabs, and that is the top of the
# face panel: 192 of its 692 px. The face's centre goes in the middle of the rest (0.639), so
# the eyes are not under the tabs.
FACE_PANEL_DOWN = (SAFE_TOP * OUT_H + FACE_H) / 2 / FACE_H
ZOOM = 1.35                         # how much tighter the zoom is
TWICE = 0.25                        # the game panel may show this much of the camera box
SLIVER = 0.06                       # ... unless a slide this short (of the frame's width) hides it
# A found camera box holds the camera's own border (the sample's is 4 px of light grey) and,
# since edges are measured on a frame shrunk to 320 px wide, up to about one shrunk pixel
# (4 px at 720p) of whatever is around it. The face panel keeps this share of the box's width
# and height away from each edge, so neither shows as a thin bright line at the panel's side.
CAMERA_INSET = 0.04

_FRAME = Rect(0.0, 0.0, 1.0, 1.0)
_UNITS = 100_000                    # single_crop wants a width and a height; only their ratio counts


def _centre(r: Rect) -> tuple[float, float]:
    return r.x + r.w / 2, r.y + r.h / 2


def _largest(shape: float, bounds: Rect) -> tuple[float, float]:
    """The biggest w, h with w / h == shape that fits in `bounds`."""
    if bounds.h * shape <= bounds.w:
        return bounds.h * shape, bounds.h
    return bounds.w, bounds.w / shape


def _place(w: float, h: float, cx: float, cy: float, bounds: Rect, down: float = 0.5) -> Rect:
    """A w by h rectangle with (cx, cy) at its middle, `down` of the way from its top, pushed
    back inside `bounds`."""
    x = min(max(cx - w / 2, bounds.x), bounds.x + bounds.w - w)
    y = min(max(cy - h * down, bounds.y), bounds.y + bounds.h - h)
    return Rect(x=x, y=y, w=w, h=h)


def _inset(camera: Rect) -> Rect:
    """The camera's picture: the box with CAMERA_INSET of it taken off every side."""
    dx, dy = camera.w * CAMERA_INSET, camera.h * CAMERA_INSET
    return Rect(x=camera.x + dx, y=camera.y + dy, w=camera.w - 2 * dx, h=camera.h - 2 * dy)


def face_panel_crop(face: Rect, camera: Rect | None, src_aspect: float) -> Rect:
    """The top panel. With a camera box, the largest panel-shaped piece of it, on the face.
    Without one, a safe crop a few face-widths wide."""
    shape = FACE_ASPECT / src_aspect
    cx, cy = _centre(face)
    if camera is not None:
        inner = _inset(camera)
        w, h = _largest(shape, inner)
        return _place(w, h, cx, cy, inner, FACE_PANEL_DOWN)
    w, h = face.w * SAFE_FACE_WIDTHS, face.w * SAFE_FACE_WIDTHS / shape
    fit_w, fit_h = _largest(shape, _FRAME)
    if w > fit_w:
        w, h = fit_w, fit_h
    return _place(w, h, cx, cy, _FRAME, FACE_PANEL_DOWN)


def _shared(a: Rect, b: Rect) -> float:
    w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return max(0.0, w) * max(0.0, h)


def game_panel_crop(camera: Rect | None, src_aspect: float) -> Rect:
    """The bottom panel: the middle of the frame, slid sideways only when it would otherwise
    show the camera box a second time."""
    w, h = _largest(GAME_ASPECT / src_aspect, _FRAME)
    centred = Rect(x=(1.0 - w) / 2, y=(1.0 - h) / 2, w=w, h=h)
    if camera is None or camera.w <= 0 or camera.h <= 0:
        return centred
    showing = _shared(centred, camera)
    if showing <= 0.0:
        return centred
    # Just clear of the camera on either side, as far as the frame allows.
    left = min(max(camera.x - w, 0.0), centred.x)
    right = max(min(camera.x + camera.w, 1.0 - w), centred.x)
    options = [Rect(x=x, y=centred.y, w=w, h=h) for x in (left, right)]
    best = min(options, key=lambda r: (round(_shared(r, camera), 9), abs(r.x - centred.x)))
    if _shared(best, camera) >= showing:
        return centred
    # A little of the camera is put up with only when hiding it would move the game a long
    # way. A short slide always goes: a sliver of the camera's edge reads as a mistake.
    if showing <= TWICE * camera.w * camera.h and abs(best.x - centred.x) > SLIVER:
        return centred
    return best


def full_face_crop(face: Rect, src_aspect: float) -> Rect:
    """One 9:16 crop that follows the face."""
    return single_crop(face, round(src_aspect * _UNITS), _UNITS)


def full_game_crop(src_aspect: float) -> Rect:
    """The centre of the game at full height, 9:16."""
    return full_face_crop(Rect(0.5, 0.5, 0.0, 0.0), src_aspect)


def zoom_crop(crop: Rect, face: Rect) -> Rect:
    """A tighter crop of the same shape on the face, inside `crop`."""
    cx, cy = _centre(face)
    return _place(crop.w / ZOOM, crop.h / ZOOM, cx, cy, crop, FACE_PANEL_DOWN)
