"""Finding the streamer's camera box: a steady small face, then the box's edges around it."""

import subprocess

import numpy as np
import pytest

from clipper import shell
from clipper.models import Rect
from streamclip.framing import camera
from streamclip.framing.camera import box_from_edges, find_camera, steady_face
from streamclip.schema import CameraBox

SPOT = Rect(0.11, 0.08, 0.07, 0.12)


class approx_rect:
    """A Rect equal to another within `abs` on every side."""

    def __init__(self, rect: Rect, abs: float):
        self.rect, self.abs = rect, abs

    def __eq__(self, other):
        return isinstance(other, Rect) and all(
            getattr(other, k) == pytest.approx(getattr(self.rect, k), abs=self.abs) for k in "xywh")

    def __repr__(self):
        return f"approx_rect({self.rect}, abs={self.abs})"


_rng = np.random.default_rng(7)


def jitter(r: Rect, by: float = 0.004) -> Rect:
    dx, dy, dw, dh = _rng.uniform(-by, by, 4)
    return Rect(r.x + dx, r.y + dy, r.w + dw, r.h + dh)


def noise_frame(seed: int, box=None, border: int = 0, size=(1280, 720)) -> np.ndarray:
    """A game that changes completely from frame to frame (16 px blocks of random colour),
    with an optional fixed camera box (x, y, w, h in pixels) that has a white border."""
    w, h = size
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8)
    frame = np.kron(blocks, np.ones((16, 16, 1), dtype=np.uint8))[:h, :w].copy()
    if box:
        x, y, bw, bh = box
        frame[y:y + bh, x:x + bw] = 255
        frame[y + border:y + bh - border, x + border:x + bw - border] = (90, 110, 130)
    return frame


def test_a_face_in_the_same_small_spot_is_steady():
    windows = [[(t / 2, jitter(SPOT), 0.0) for t in range(40)] for _ in range(6)]
    assert steady_face(windows) == approx_rect(SPOT, abs=0.01)


def test_a_big_face_is_not_a_camera_box():
    big = Rect(0.3, 0.1, 0.35, 0.6)
    assert steady_face([[(t / 2, big, 0.0) for t in range(40)]] * 6) is None


def test_a_face_that_wanders_is_not_steady():
    assert steady_face([[(t / 2, Rect(0.02 * t, 0.1, 0.07, 0.12), 0.0) for t in range(40)]] * 6) is None


def test_a_face_seen_in_too_few_windows_is_not_steady():
    spot = [(t / 2, SPOT, 0.0) for t in range(40)]
    assert steady_face([spot, [], [], [], [], []]) is None


def test_no_faces_at_all_is_not_steady():
    assert steady_face([]) is None and steady_face([[], []]) is None


def test_a_passing_second_face_does_not_unseat_the_steady_one():
    window = [(t / 2, jitter(SPOT), 0.0) for t in range(40)]
    window += [(t / 2, Rect(0.5 + 0.01 * t, 0.5, 0.05, 0.09), 0.0) for t in range(40)]   # a face in the game
    assert steady_face([window] * 6) == approx_rect(SPOT, abs=0.01)


def test_edges_of_a_fixed_box_are_found_over_a_changing_game():
    frames = [noise_frame(seed=i, box=(26, 22, 320, 180), border=3) for i in range(24)]     # 1280x720
    rect, confidence = box_from_edges(frames, SPOT)
    assert rect == approx_rect(Rect(26 / 1280, 22 / 720, 320 / 1280, 180 / 720), abs=0.01)
    assert confidence > 0.5


def test_a_box_flush_in_the_corner_is_found_from_its_two_inner_edges():
    frames = [noise_frame(seed=i, box=(0, 0, 320, 180), border=3) for i in range(24)]
    rect, _ = box_from_edges(frames, Rect(0.09, 0.05, 0.07, 0.12))
    assert rect == approx_rect(Rect(0.0, 0.0, 0.25, 0.25), abs=0.01)


def test_a_camera_with_no_drawn_border_is_found_where_its_picture_ends():
    import cv2
    room = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 256, (90, 160, 3), dtype=np.uint8), (0, 0), 6)
    room = cv2.normalize(room, None, 30, 220, cv2.NORM_MINMAX)        # a still, soft picture
    frames = []
    for i in range(60):
        frame = cv2.GaussianBlur(noise_frame(seed=i, size=(640, 360)), (0, 0), 2)
        frame[180:270, 240:400] = room
        frames.append(frame)
    rect, confidence = box_from_edges(frames, Rect(0.47, 0.55, 0.055, 0.125))
    assert rect == approx_rect(Rect(240 / 640, 180 / 360, 0.25, 0.25), abs=0.01) and confidence > 0.5


def test_no_clean_edge_gives_none():
    assert box_from_edges([noise_frame(seed=i) for i in range(24)], SPOT) is None
    assert box_from_edges([], SPOT) is None


def test_a_box_of_the_wrong_size_for_the_face_is_not_the_camera():
    frames = [noise_frame(seed=i, box=(26, 22, 960, 540), border=3) for i in range(24)]     # 13 faces wide
    assert box_from_edges(frames, SPOT) is None


def test_the_fixture_camera_box_is_found(synthetic_stream):
    face = lambda video, start, end, fps=2.0: [(start + i / 2, Rect(0.11, 0.09, 0.06, 0.11), 0.0) for i in range(20)]
    box = find_camera([synthetic_stream] * 4, detect=face)
    assert box.rect == approx_rect(Rect(24 / 1280, 24 / 720, 320 / 1280, 180 / 720), abs=0.02)
    assert box.confidence > 0.3


def test_no_face_anywhere_means_no_camera(synthetic_stream):
    assert find_camera([synthetic_stream] * 4, detect=lambda *a, **k: []) is None
    assert find_camera([], detect=lambda *a, **k: []) is None


def test_faces_are_sampled_twice_a_second_from_the_start_of_each_window(synthetic_stream):
    calls = []

    def spy(video, start, end, fps=10.0):
        calls.append((video, start, end, fps))
        return []

    find_camera([synthetic_stream] * 3, detect=spy)
    assert calls == [(synthetic_stream, 0.0, 20.0, 2.0)] * 3


def test_the_real_detector_is_looked_up_when_called(monkeypatch, synthetic_stream):
    from streamclip.framing import smallfaces
    monkeypatch.setattr(smallfaces, "detect_faces", lambda *a, **k: [])
    assert find_camera([synthetic_stream]) is None


FIXTURE_FACE = Rect(0.11, 0.09, 0.06, 0.11)             # inside the test stream's camera box


def test_a_camera_too_small_for_the_whole_frame_is_searched_for_in_parts(synthetic_stream):
    """A corner camera's face is missed in the whole frame: found on a real broadcast, where
    every clip then came out without the streamer."""
    calls = []

    def search(video, start, end, fps=10.0):
        calls.append((start, end, fps))
        return [(start + 2 * i, FIXTURE_FACE, 0.0) for i in range(10)]

    box = find_camera([synthetic_stream] * 4, detect=lambda *a, **k: [], search=search)
    assert box.rect == approx_rect(Rect(24 / 1280, 24 / 720, 320 / 1280, 180 / 720), abs=0.02)
    assert calls == [(0.0, 20.0, 0.5)] * 4                # a look every two seconds: each costs 26


def test_the_search_is_not_needed_when_the_whole_frame_shows_the_face(synthetic_stream):
    def search(*a, **k):
        raise AssertionError("searched although the face was already found")

    face = lambda video, start, end, fps=2.0: [(start + i / 2, FIXTURE_FACE, 0.0) for i in range(20)]
    assert find_camera([synthetic_stream] * 4, detect=face, search=search) is not None


def test_the_search_looks_at_twelve_windows_at_most_spread_across_the_stream(tmp_path, synthetic_stream):
    windows = [tmp_path / f"{n:02d}.mp4" for n in range(30)]
    seen = []
    find_camera(windows, detect=lambda *a, **k: [], search=lambda video, *a, **k: seen.append(video) or [])
    assert len(seen) == 12 and seen[0] == windows[0] and seen[-1] == windows[-1]
    assert seen == sorted(seen)


def test_a_caller_s_own_detector_is_not_followed_by_the_real_search(monkeypatch, synthetic_stream):
    from streamclip.framing import smallfaces
    monkeypatch.setattr(smallfaces, "detect_faces", lambda *a, **k: pytest.fail("the real search ran"))
    assert find_camera([synthetic_stream] * 2, detect=lambda *a, **k: []) is None


def test_the_real_search_is_looked_up_when_called_and_uses_tiles(monkeypatch, synthetic_stream):
    from streamclip.framing import smallfaces

    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        if not tiles:
            return []                                     # the whole frame shows nothing
        assert tiles == smallfaces.TILES
        return [(start + 2 * i, FIXTURE_FACE, 0.0) for i in range(10)]

    monkeypatch.setattr(smallfaces, "detect_faces", detect)
    assert find_camera([synthetic_stream] * 4) is not None


def test_a_steady_face_with_no_edges_gets_the_safe_box(tmp_path):
    video = tmp_path / "noise.mp4"                    # a plain moving-noise video
    try:
        subprocess.run([shell.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i",
                        "nullsrc=s=640x360:r=10,geq=lum='random(1)*255':cb=128:cr=128",
                        "-t", "6", "-pix_fmt", "yuv420p", str(video)], check=True, capture_output=True)
    except OSError as exc:
        pytest.skip(f"ffmpeg is needed to build the noise video: {exc}")
    face = Rect(0.50, 0.40, 0.07, 0.12)
    box = find_camera([video] * 4, detect=lambda *a, **k: [(i / 2, face, 0.0) for i in range(12)])
    assert isinstance(box, CameraBox) and box.confidence == 0.3
    assert box.rect.w == pytest.approx(face.w * 3.2)
    assert (box.rect.w * 16 / 9) / box.rect.h == pytest.approx(16 / 9)       # 16:9 in pixels
    assert box.rect.x < face.x and box.rect.x + box.rect.w > face.x + face.w
    assert box.rect.y < face.y and box.rect.y + box.rect.h > face.y + face.h


def test_the_safe_box_stays_inside_the_frame():
    rect = camera.safe_box(Rect(0.94, 0.0, 0.06, 0.1), 16 / 9)
    assert rect.x + rect.w == pytest.approx(1.0) and rect.y == 0.0
