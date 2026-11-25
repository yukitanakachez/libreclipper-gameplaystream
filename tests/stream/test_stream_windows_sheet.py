"""Lane D, task 2: one contact sheet of 15 frames per window, for the judge to look at."""

import subprocess

import numpy as np
import pytest
from PIL import Image

from clipper import shell
from streamclip.windows import sheet
from streamclip.windows.sheet import contact_sheet

CELL_W, CELL_H = 384, 216


def cell(im, i):
    x, y = (i % 5) * CELL_W, (i // 5) * CELL_H
    return np.asarray(im.convert("RGB").crop((x, y + 30, x + CELL_W, y + CELL_H)), dtype=np.float32)


def cut(src, dest, args):
    subprocess.run([shell.ffmpeg(), "-v", "error", "-y", *args, str(dest)], check=True,
                   capture_output=True)
    return dest


@pytest.fixture(scope="module")
def short_video(synthetic_stream, tmp_path_factory):
    """Four seconds of the fixture, picture and sound."""
    return cut(synthetic_stream, tmp_path_factory.mktemp("sheet") / "short.mp4",
               ["-ss", "20", "-t", "4", "-i", str(synthetic_stream), "-c:v", "libx264",
                "-preset", "ultrafast", "-c:a", "aac"])


def test_sheet_is_a_five_by_three_grid(synthetic_stream, tmp_path):
    out = contact_sheet(synthetic_stream, tmp_path / "sheet.jpg")
    assert out == tmp_path / "sheet.jpg"
    with Image.open(out) as im:
        assert im.format == "JPEG" and im.size == (5 * 384, 3 * 216)
        assert all(cell(im, i).mean() > 20 for i in range(15))     # every cell has a picture
    assert out.stat().st_size < 600_000


def test_frame_times_are_spread_across_the_video():
    assert sheet.frame_times(90.0, 15) == [3.0 + 6.0 * i for i in range(15)]
    assert sheet.frame_times(0.0, 15) == []


def test_each_cell_is_labelled_with_its_time():
    assert sheet.label(3.0) == "3s" and sheet.label(45.4) == "45s" and sheet.label(87.0) == "87s"


def test_cells_show_the_frame_from_their_own_time(synthetic_stream, tmp_path):
    """The fixture's picture freezes from 70 s to 80 s. Cut 65-75 s: the cells at 0.3-4.3 s
    move, the cells at 5.7-9.7 s all show the one still frame."""
    video = cut(synthetic_stream, tmp_path / "freeze.mp4",
                ["-ss", "65", "-t", "10", "-i", str(synthetic_stream), "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "12", "-c:a", "aac"])
    with Image.open(contact_sheet(video, tmp_path / "sheet.jpg")) as im:
        moving = [cell(im, i) for i in range(0, 7)]
        frozen = [cell(im, i) for i in range(9, 15)]
    assert all(np.abs(a - b).mean() > 5 for a, b in zip(moving, moving[1:]))
    assert all(np.abs(a - b).mean() < 2 for a, b in zip(frozen, frozen[1:]))


def test_a_short_video_still_makes_a_sheet(short_video, tmp_path):
    out = contact_sheet(short_video, tmp_path / "sheet.jpg")
    with Image.open(out) as im:
        assert im.size == (5 * 384, 3 * 216)
        assert all(cell(im, i).mean() > 20 for i in range(15))


def test_cells_past_the_last_frame_are_black(synthetic_stream, tmp_path):
    """Picture for 4 s, sound for 8 s: the cells after the picture ends stay black."""
    video = cut(synthetic_stream, tmp_path / "early-end.mp4",
                ["-ss", "20", "-t", "8", "-i", str(synthetic_stream),
                 "-filter_complex", "[0:v]trim=end=4[v]", "-map", "[v]", "-map", "0:a",
                 "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac"])
    out = contact_sheet(video, tmp_path / "sheet.jpg")
    with Image.open(out) as im:
        assert im.size == (5 * 384, 3 * 216)
        assert cell(im, 0).mean() > 20
        assert cell(im, 14).mean() < 2


def test_a_file_with_no_picture_is_refused(tmp_path, synthetic_stream):
    sound = cut(synthetic_stream, tmp_path / "sound.m4a",
                ["-t", "5", "-i", str(synthetic_stream), "-vn", "-c:a", "aac"])
    with pytest.raises(ValueError, match="no frames"):
        contact_sheet(sound, tmp_path / "sheet.jpg")
    assert not (tmp_path / "sheet.jpg").exists()
