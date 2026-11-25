"""Which of the three layouts each shot of a clip gets, and what it crops.

The framing only changes where the source itself cuts, so a change never looks like a mistake.
Times in the result are source-clip time: 0 is `pick.start`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from clipper.models import Rect
from streamclip.framing import crops
from streamclip.schema import CameraBox, Layout, Pick, Shot

FACE_FPS = 5.0                  # faces sampled per second across the pick
SEARCH_LOOKS = 12               # frames searched in parts when a clip shows no face where expected
MIN_COVERAGE = 0.5              # the main face is in at least this share of a shot's samples
ON_CAMERA = 0.4                 # or, inside the stream's own camera box, this share
BIG_FACE = 0.30                 # a face this tall (fraction of the frame) fills the frame itself
MIN_SHOT = 0.5                  # a shorter shot takes its longer neighbour's layout
HIDDEN = 4.0                    # a face out of sight for less than this is still on camera
SAME_CROP = 0.02                # neighbours whose crops differ by less are one shot
DEFAULT_ASPECT = 16 / 9


def _aspect(video: Path) -> float:
    """The source's width / height, read from the window video."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        cap.release()
    return w / h if w > 0 and h > 0 else DEFAULT_ASPECT


def _holds(box: Rect, face: Rect) -> bool:
    """Whether the middle of `face` is inside `box`."""
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    return box.x <= cx <= box.x + box.w and box.y <= cy <= box.y + box.h


def _main_face(detections: list, start: float, end: float, camera: CameraBox | None = None,
               in_box: float = MIN_COVERAGE) -> Rect | None:
    """The face seen most in [start, end), if it is there for at least half of it.

    Every face inside the camera box is the streamer. One who leans from side to side there is
    seen as several faces, each for under half the shot, so those are counted together, and
    they need only `in_box` of the shot.
    """
    from clipper.reframe import faces

    def most_seen(found: list) -> faces.FaceTrack | None:
        tracks = faces.tracks_in(found, start, end)
        return max(tracks, key=lambda track: len(track.samples)) if tracks else None

    enough = MIN_COVERAGE * max(1, round((end - start) * FACE_FPS))
    main = most_seen(detections)
    if main is not None and len({sample.t for sample in main.samples}) >= enough:
        return main.median_box()
    if camera is not None:
        on_camera = [d for d in detections if start <= d[0] < end and _holds(camera.rect, d[1])]
        if on_camera and len({t for t, _, _ in on_camera}) >= in_box * max(1, round((end - start) * FACE_FPS)):
            return most_seen(on_camera).median_box()
    return None


def _tidy(r: Rect | None) -> Rect | None:
    return None if r is None else Rect(*(round(getattr(r, k), 5) for k in "xywh"))


def _frame(start: float, end: float, face: Rect | None, camera: CameraBox | None, aspect: float) -> Shot:
    if face is None:
        return Shot(start, end, "game", face=None, game=_tidy(crops.full_game_crop(aspect)), zoom=None)
    if face.h >= BIG_FACE:
        crop = crops.full_face_crop(face, aspect)
        return Shot(start, end, "face", face=_tidy(crop), game=None, zoom=_tidy(crops.zoom_crop(crop, face)))
    # The camera box is trusted only when the face is in it. Otherwise the finder was wrong for
    # this shot, and a safe crop around the face is used.
    box = camera.rect if camera is not None and _holds(camera.rect, face) else None
    crop = crops.face_panel_crop(face, box, aspect)
    return Shot(start, end, "split", face=_tidy(crop), game=_tidy(crops.game_panel_crop(box, aspect)),
                zoom=_tidy(crops.zoom_crop(crop, face)))


def _length(shot: Shot) -> float:
    return shot.end - shot.start


def _apart(a: Rect | None, b: Rect | None) -> float:
    if a is None or b is None:
        return 0.0 if a is b else float("inf")
    return max(abs(getattr(a, k) - getattr(b, k)) for k in "xywh")


def _absorb(shots: list[Shot], keep: int, drop: int) -> None:
    """Shot `keep` takes over the time of its neighbour `drop` and keeps its own layout."""
    a, b = shots[keep], shots[drop]
    shots[keep] = replace(a, start=min(a.start, b.start), end=max(a.end, b.end))
    del shots[drop]


def _settle(shots: list[Shot]) -> list[Shot]:
    """Flickers take a neighbour's layout, then neighbours with the same layout are joined."""
    shots = list(shots)
    while len(shots) > 1:
        i = min(range(len(shots)), key=lambda n: _length(shots[n]))
        if _length(shots[i]) >= MIN_SHOT:
            break
        around = [n for n in (i - 1, i + 1) if 0 <= n < len(shots)]
        _absorb(shots, max(around, key=lambda n: _length(shots[n])), i)
    i = 0
    while i + 1 < len(shots):
        a, b = shots[i], shots[i + 1]
        if a.kind == b.kind and max(_apart(a.face, b.face), _apart(a.game, b.game)) < SAME_CROP:
            if _length(a) >= _length(b):
                _absorb(shots, i, i + 1)
            else:
                _absorb(shots, i + 1, i)
        else:
            i += 1
    return shots


def _look(window: Path, a: float, b: float, camera: CameraBox | None,
          aspect: float) -> tuple[list, CameraBox | None]:
    """The faces across one clip, and the camera box to frame it with.

    The whole frame is always looked at, for a streamer who fills it, and so is the stream's
    camera box, where the face is too small for the whole frame to show it. When that leaves
    most of the clip without a face (no camera was found for the stream, or it sits somewhere
    else in this game), the clip is searched in parts on its own, and a face that stays in one
    spot gets a safe box around it.
    """
    from streamclip.framing import camera as finder
    from streamclip.framing import smallfaces

    regions = [camera.rect] if camera is not None else []
    detections = smallfaces.detect_faces(window, a, b, fps=FACE_FPS, regions=regions)
    expected = max(1, round((b - a) * FACE_FPS))
    if len({t for t, _, _ in detections}) >= MIN_COVERAGE * expected:
        return detections, camera
    sparse = smallfaces.detect_faces(window, a, b, fps=SEARCH_LOOKS / (b - a), tiles=smallfaces.TILES)
    face = finder.steady_face([sparse])
    if face is None:
        return detections, camera
    local = CameraBox(rect=finder.safe_box(face, aspect), confidence=finder.SAFE_CONFIDENCE)
    return smallfaces.detect_faces(window, a, b, fps=FACE_FPS, regions=[local.rect]), local


def _hold(shots: list[Shot]) -> list[Shot]:
    """A short game shot beside a split one stays split. A hand over the face or a turn of the
    head hides the face from the finder, but the camera is still on screen."""
    shots = list(shots)
    i = 0
    while i < len(shots):
        beside = [n for n in (i - 1, i + 1) if 0 <= n < len(shots) and shots[n].kind == "split"]
        if shots[i].kind == "game" and _length(shots[i]) < HIDDEN and beside:
            _absorb(shots, max(beside, key=lambda n: _length(shots[n])), i)
            i = max(0, i - 1)
        else:
            i += 1
    return shots


def plan_layout(window: Path, window_start: float, pick: Pick, camera: CameraBox | None,
                detect=None, shots=None, clip_video: Path | None = None) -> Layout:
    """The layout of one clip, worked out on its candidate's low-resolution window.

    `window_start` is the stream second at the window's t=0. `shots` defaults to the podcast
    clipper's shot detection. `detect` defaults to the real face model, which looks inside the
    camera box as well as at the whole frame and searches the clip when it has to (`_look`);
    a `detect` passed in is asked once, for whole frames.

    `clip_video` is the pick's own stretch at full size (t=0 at `pick.start`). When it is there
    the real face model reads it instead of the window: a corner camera's face is 30 to 36 px
    tall in the 360p window and missed in most samples. The shot cuts still come from the
    window, which is cheap to scan, and the faces are moved onto its clock.
    """
    if shots is None:
        from clipper.reframe import shots as shot_detection      # imports scenedetect
        shots = shot_detection.detect_shots

    a, b = pick.start - window_start, pick.end - window_start    # the window's own clock
    if b <= a:
        return Layout(clip=pick.rank, shots=[])
    aspect = _aspect(window)
    stream_camera = camera
    if detect is None:
        if clip_video is not None and Path(clip_video).exists():
            found, camera = _look(Path(clip_video), 0.0, b - a, camera, aspect)
            detections = [(t + a, box, mouth) for t, box, mouth in found]
        else:
            detections, camera = _look(window, a, b, camera, aspect)
    else:
        detections = detect(window, a, b, fps=FACE_FPS)

    # Only the cut points are taken from the shot detector, so the shots always cover the
    # whole pick with no gaps, whatever it returns.
    cuts = sorted({a, b} | {t for shot in shots(window, a, b) for t in shot[:2] if a < t < b})
    # The stream's own box was seen steady across the stream, so the camera is on screen even
    # where the face model misses the face or the streamer looks away: a face in two fifths of
    # the shot is enough there (run 10's clip 4: 34 of 71 samples). A box made from this clip's
    # own search is a guess, and needs half, like a face with no box.
    in_box = ON_CAMERA if camera is not None and camera is stream_camera else MIN_COVERAGE
    planned = []
    for start, end in zip(cuts, cuts[1:]):
        rel_start, rel_end = round(start - a, 3), round(end - a, 3)
        if rel_end > rel_start:
            face = _main_face(detections, start, end, camera, in_box)
            planned.append(_frame(rel_start, rel_end, face, camera, aspect))
    return Layout(clip=pick.rank, shots=_settle(_hold(_settle(planned))))
