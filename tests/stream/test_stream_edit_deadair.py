"""Dead air: a stretch is cut only when nobody speaks, the sound is quiet and the picture is still."""

from pathlib import Path

import numpy as np
import pytest

from clipper import shell
from streamclip import audio
from streamclip.edit import deadair
from streamclip.schema import Word


def cut(stream: Path, start: float, end: float, folder: Path) -> Path:
    """That stretch of the test stream as its own file, where t=0 is `start`."""
    dest = Path(folder) / f"cut-{start:g}-{end:g}.mp4"
    if not dest.exists():
        shell.run([shell.ffmpeg(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                   "-i", str(stream), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(dest)])
    return dest


@pytest.fixture(scope="module")
def clip(synthetic_stream, tmp_path_factory):
    """clip(start, end) -> (pcm, motion) for that stretch of the test stream."""
    folder = tmp_path_factory.mktemp("deadair")
    made = {}

    def make(start: float, end: float):
        if (start, end) not in made:
            path = cut(synthetic_stream, start, end, folder)
            made[start, end] = (audio.read_pcm(path), deadair.motion_series(path))
        return made[start, end]
    return make


def noise(seconds: float, db: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    samples = rng.uniform(-1.0, 1.0, int(round(seconds * 16000))).astype(np.float32)
    return samples * np.float32(10 ** (db / 20) * np.sqrt(3))        # uniform noise at `db` RMS


def loud_noise(seconds: float) -> np.ndarray:
    return noise(seconds, -12.0)


def silent_and_still(seconds: float, inside_seconds: float):
    """Noise and movement throughout, but for `seconds` in the middle: silence, nothing moving."""
    pcm = noise(inside_seconds, -30.0)
    motion = np.full(int(inside_seconds * 10), 8.0, dtype=np.float32)
    a = 3.0
    pcm[int(a * 16000):int((a + seconds) * 16000)] = 0.0
    motion[int(a * 10) - 2:int((a + seconds) * 10) + 3] = 0.0   # still a little longer than it is silent
    return pcm, motion


def cut_spans(keep, length):                         # the gaps between kept intervals
    edges = [0.0] + [x for a, b in keep for x in (a, b)] + [length]
    return [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2) if edges[i + 1] - edges[i] > 1e-6]


def test_motion_is_zero_on_the_frozen_stretch(synthetic_stream, tmp_path):
    motion = deadair.motion_series(cut(synthetic_stream, 60, 90, tmp_path))       # 10 samples a second
    assert len(motion) == pytest.approx(300, abs=3)
    assert motion[0] == 0.0
    assert motion[120:190].max() < 0.5 and motion[20:80].mean() > 2.0


def test_silent_and_still_with_no_words_is_cut_to_a_breath(clip):                # Review Focus 1
    pcm, motion = clip(60.0, 90.0)
    keep = deadair.keep_intervals([], 60.0, 90.0, pcm, 16000, motion, 10.0, [], 30.0)
    (gap,) = cut_spans(keep, 30.0)
    assert gap[0] == pytest.approx(10.05, abs=0.1) and gap[1] == pytest.approx(19.95, abs=0.1)
    assert sum(b - a for a, b in keep) == pytest.approx(20.1, abs=0.15)


def test_a_protected_stretch_is_never_cut(clip):
    pcm, motion = clip(60.0, 90.0)
    keep = deadair.keep_intervals([], 60.0, 90.0, pcm, 16000, motion, 10.0, [(69.0, 81.0)], 30.0)
    assert keep == [(0.0, 30.0)]


def test_a_stretch_protected_in_part_is_cut_only_outside_it(clip):
    pcm, motion = clip(60.0, 90.0)
    keep = deadair.keep_intervals([], 60.0, 90.0, pcm, 16000, motion, 10.0, [(69.0, 75.0)], 30.0)
    (gap,) = cut_spans(keep, 30.0)
    assert gap[0] >= 15.0 and gap[1] == pytest.approx(19.95, abs=0.1)


def test_quiet_but_moving_stays(clip):               # 40-48 s: quiet noise, moving picture
    pcm, motion = clip(36.0, 48.0)
    assert deadair.keep_intervals([], 36.0, 48.0, pcm, 16000, motion, 10.0, [], 30.0) == [(0.0, 12.0)]


def test_still_but_loud_stays():
    pcm = loud_noise(seconds=10)
    assert deadair.keep_intervals([], 0.0, 10.0, pcm, 16000, np.zeros(100), 10.0, [], 30.0) == [(0.0, 10.0)]


def test_words_are_never_cut_into(clip):
    pcm, motion = clip(60.0, 90.0)
    words = [Word(0, "wait", 74.0, 74.6)]             # someone speaks inside the dead stretch
    keep = deadair.keep_intervals(words, 60.0, 90.0, pcm, 16000, motion, 10.0, [], 30.0)
    assert any(a <= 13.95 and b >= 14.65 for a, b in keep) and len(cut_spans(keep, 30.0)) == 2


def test_quiet_is_measured_against_the_speech():
    """Speech at -20 dB, then a still stretch at -30 dB (kept: only 10 under) and one at -50 dB (cut)."""
    pcm = np.concatenate([noise(4, -20.0), noise(3, -30.0), noise(3, -50.0), noise(2, -20.0)])
    words = [Word(0, "hello", 0.2, 3.8), Word(1, "again", 10.2, 11.8)]
    keep = deadair.keep_intervals(words, 0.0, 12.0, pcm, 16000, np.zeros(120), 10.0, [], 30.0)
    (gap,) = cut_spans(keep, 12.0)
    assert gap[0] == pytest.approx(7.05, abs=0.04) and gap[1] == pytest.approx(9.95, abs=0.04)


def test_half_a_second_or_less_is_left_alone():
    pcm, motion = silent_and_still(seconds=0.5, inside_seconds=6)
    assert deadair.keep_intervals([], 0.0, 6.0, pcm, 16000, motion, 10.0, [], 30.0) == [(0.0, 6.0)]


def test_just_over_half_a_second_is_cut():
    pcm, motion = silent_and_still(seconds=0.6, inside_seconds=6)
    keep = deadair.keep_intervals([], 0.0, 6.0, pcm, 16000, motion, 10.0, [], 30.0)
    assert sum(b - a for a, b in keep) == pytest.approx(5.5, abs=0.07)


def test_the_threshold_can_be_changed(clip):
    pcm, motion = clip(60.0, 90.0)
    assert deadair.keep_intervals([], 60.0, 90.0, pcm, 16000, motion, 10.0, [], 30.0, dead_air=15.0) == [(0.0, 30.0)]


def test_edges_land_on_frames(clip):
    pcm, motion = clip(60.0, 90.0)
    for a, b in deadair.keep_intervals([], 60.0, 90.0, pcm, 16000, motion, 10.0, [], 30.0):
        assert abs(a * 30 - round(a * 30)) < 1e-6 and abs(b * 30 - round(b * 30)) < 1e-6


def test_missing_sound_or_picture_is_never_dead():
    """A clip whose sound or movement could not be measured is kept whole, not cut blind."""
    silence = np.zeros(16000 * 6, np.float32)
    assert deadair.keep_intervals([], 0.0, 6.0, silence, 16000, np.zeros(0), 10.0, [], 30.0) == [(0.0, 6.0)]
    assert deadair.keep_intervals([], 0.0, 6.0, np.zeros(0, np.float32), 16000, np.zeros(60), 10.0, [], 30.0) == [(0.0, 6.0)]
