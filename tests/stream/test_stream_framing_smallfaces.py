"""Faces that are small in the frame: a streamer's camera in a corner.

The face model misses a face that is a small share of the picture it is shown, so the frame is
also looked at in parts. Found on a real broadcast whose camera sat in the bottom right corner:
no clip showed the streamer at all.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipper import shell
from clipper.models import Rect
from streamclip.framing import smallfaces

FACE = Rect(0.85, 0.80, 0.06, 0.11)             # a corner camera's face: 6% of the frame wide
SIZE = (360, 640)                                # rows, columns


def picture(face: Rect = FACE) -> np.ndarray:
    """A black frame with the face as a white block."""
    h, w = SIZE
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[round(face.y * h):round((face.y + face.h) * h), round(face.x * w):round((face.x + face.w) * w)] = 255
    return rgb


def needs_a_big_face(rgb: np.ndarray) -> list[tuple[Rect, float]]:
    """Stands in for the face model: it sees the white block only when the block is at least a
    tenth of the picture wide, and reports whatever part of it is in the picture."""
    ys, xs = np.nonzero(rgb[:, :, 0])
    if not len(xs):
        return []
    h, w = rgb.shape[:2]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if (x1 - x0) / w < 0.10:
        return []
    return [(Rect(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h), 0.25)]


@pytest.fixture
def four_frames(monkeypatch):
    """Reading a video gives four of `picture()` half a second apart; returns a way to change the face."""
    shown = {"face": FACE}

    def sampled(video, start, end, fps=10.0):
        for i in range(4):
            yield start + i / fps, picture(shown["face"])

    monkeypatch.setattr("clipper.reframe.faces.iter_sampled_frames", sampled)
    return shown


def close(a: Rect, b: Rect, tolerance: float = 0.01) -> bool:
    return all(abs(getattr(a, k) - getattr(b, k)) <= tolerance for k in "xywh")


def test_the_whole_frame_is_always_looked_at_first():
    assert smallfaces.views() == [Rect(0.0, 0.0, 1.0, 1.0)]
    assert smallfaces.views(3, [FACE])[0] == Rect(0.0, 0.0, 1.0, 1.0)


def test_tiles_are_a_third_of_the_frame_and_step_by_half_a_tile():
    tiles = smallfaces.views(3)[1:]
    assert len(tiles) == 25
    assert all(t.w == pytest.approx(1 / 3) and t.h == pytest.approx(1 / 3) for t in tiles)
    assert min(t.x for t in tiles) == 0.0 and min(t.y for t in tiles) == 0.0
    assert max(t.x + t.w for t in tiles) == pytest.approx(1.0)
    assert max(t.y + t.h for t in tiles) == pytest.approx(1.0)


def test_a_region_is_looked_at_a_little_wider_and_never_outside_the_frame():
    whole, region = smallfaces.views(0, [Rect(0.60, 0.50, 0.20, 0.20)])
    assert close(region, Rect(0.57, 0.47, 0.26, 0.26), 1e-9)
    _, corner = smallfaces.views(0, [Rect(0.78, 0.70, 0.22, 0.30)])
    assert corner.x < 0.78 and corner.y < 0.70
    assert corner.x + corner.w == pytest.approx(1.0) and corner.y + corner.h == pytest.approx(1.0)


def test_the_whole_frame_alone_misses_a_corner_camera(four_frames):
    assert smallfaces.detect_faces("v.mp4", 0.0, 2.0, fps=2.0, look=needs_a_big_face) == []


def test_tiles_find_it_once_per_moment_in_whole_frame_fractions(four_frames):
    found = smallfaces.detect_faces("v.mp4", 0.0, 2.0, fps=2.0, tiles=3, look=needs_a_big_face)
    assert [t for t, _, _ in found] == [0.0, 0.5, 1.0, 1.5]          # several tiles saw it: one answer
    assert all(close(box, FACE) and mouth == 0.25 for _, box, mouth in found)


def test_half_a_face_at_a_tile_s_cut_edge_is_not_an_answer(four_frames):
    # The tile that ends at 5/6 of the height holds only the top of this face; the whole of it
    # is in the next tile, and that is the one reported.
    found = smallfaces.detect_faces("v.mp4", 0.0, 2.0, fps=2.0, tiles=3, look=needs_a_big_face)
    assert all(box.h == pytest.approx(FACE.h, abs=0.01) for _, box, _ in found)


def test_a_face_against_the_frame_s_own_edge_is_kept(four_frames):
    four_frames["face"] = Rect(0.94, 0.89, 0.06, 0.11)
    found = smallfaces.detect_faces("v.mp4", 0.0, 2.0, fps=2.0, tiles=3, look=needs_a_big_face)
    assert len(found) == 4 and close(found[0][1], four_frames["face"])


def test_a_region_finds_it_with_one_extra_look(four_frames):
    looks = []

    def counting(rgb):
        looks.append(rgb.shape[:2])
        return needs_a_big_face(rgb)

    camera = Rect(0.775, 0.71, 0.225, 0.29)
    found = smallfaces.detect_faces("v.mp4", 0.0, 2.0, fps=2.0, regions=[camera], look=counting)
    assert len(found) == 4 and all(close(box, FACE) for _, box, _ in found)
    assert len(looks) == 8                                       # the whole frame and the region, per frame


def test_two_faces_stay_two(monkeypatch):
    other = Rect(0.10, 0.10, 0.06, 0.11)

    def sampled(video, start, end, fps=10.0):
        frame = picture()
        frame[36:76, 64:102] = 255                              # a second block, top left
        yield start, frame

    def two(rgb):
        """Sees every white block that is big enough, each on its own."""
        import cv2
        count, _, stats, _ = cv2.connectedComponentsWithStats((rgb[:, :, 0] > 0).astype(np.uint8))
        h, w = rgb.shape[:2]
        return [(Rect(x / w, y / h, bw / w, bh / h), 0.0) for x, y, bw, bh, _ in stats[1:count] if bw / w >= 0.10]

    monkeypatch.setattr("clipper.reframe.faces.iter_sampled_frames", sampled)
    found = smallfaces.detect_faces("v.mp4", 0.0, 1.0, fps=1.0, tiles=3, look=two)
    assert sorted(box.x for _, box, _ in found) == pytest.approx([other.x, FACE.x], abs=0.01)


def test_the_real_model_finds_a_corner_camera_only_when_it_looks_at_parts(tmp_path):
    """The sample stream's face, shrunk into the bottom right corner of a 1920x1080 frame."""
    from clipper.reframe import faces

    sample = Path(__file__).resolve().parents[2] / "agents" / "gameplay-clipper" / "tests" / "sample-stream.mp4"
    if not sample.exists() or not (shell.project_home() / "assets" / faces.MODEL_FILE).exists():
        pytest.skip("needs the sample stream and the face model")
    video = tmp_path / "corner.mp4"
    try:
        subprocess.run([shell.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=10",
                        "-ss", "30", "-i", str(sample), "-filter_complex",
                        "[1:v]crop=320:256:28:28,scale=224:180[cam];[0:v][cam]overlay=W-w:H-h",
                        "-t", "4", "-r", "10", "-pix_fmt", "yuv420p", "-an", str(video)],
                       check=True, capture_output=True)
    except OSError as exc:
        pytest.skip(f"ffmpeg is needed to build the corner-camera video: {exc}")

    whole = smallfaces.detect_faces(video, 0.0, 4.0, fps=2.0)
    parts = smallfaces.detect_faces(video, 0.0, 4.0, fps=2.0, tiles=smallfaces.TILES)
    in_corner = [box for _, box, _ in parts if box.x > 0.88 and box.y > 0.83]
    assert len({t for t, _, _ in parts}) >= 6                   # of 8 moments
    assert len(in_corner) >= 6 and all(box.w < 0.10 for box in in_corner)
    assert len({t for t, _, _ in whole}) < len({t for t, _, _ in parts})
