from streamclip import settings as s
from streamclip.errors import NoVideo, Refused, StageError


def test_spec_values():
    assert (s.OUT_W, s.OUT_H) == (1080, 1920)
    assert (s.FACE_H, s.GAME_H) == (692, 1228) and s.FACE_H + s.GAME_H == s.OUT_H
    # yuv420p stores colour for pairs of rows: stacking an odd-height panel overruns ffmpeg's
    # vstack buffer, which glibc aborts on (Starwish) and macOS lets pass.
    assert s.FACE_H % 2 == 0 and s.GAME_H % 2 == 0
    assert s.SPEED == 1.3 and s.DEAD_AIR == 0.5 and s.BREATH == 0.1
    assert (s.MIN_FINISHED, s.MAX_FINISHED) == (15.0, 40.0)
    assert (s.DEFAULT_CLIPS, s.MIN_CLIPS, s.MAX_CLIPS, s.FLOOR) == (6, 1, 10, 5)
    assert (s.SHORT_STREAM, s.SHORT_STREAM_PER_CLIP, s.MAX_STREAM) == (45 * 60, 8 * 60, 6 * 3600)
    assert (s.PEAK_GAP, s.PEAKS_PER_CLIP, s.WINDOW_BEFORE, s.WINDOW_AFTER) == (45.0, 4, 60.0, 30.0)
    assert s.ACCENT == "#D7FF3E" and s.TILT == 4 and s.SHOUT_MAX == 165 and s.TITLE_SECONDS == 3.0
    assert (s.MAX_SOUNDS, s.SOUND_GAP) == (5, 4.0)
    assert (s.SAFE_TOP, s.SAFE_BOTTOM, s.SAFE_RIGHT, s.CAPTION_Y) == (0.10, 0.20, 0.12, 0.71)
    assert s.LOUDNESS == "I=-14:TP=-1:LRA=11"


def test_no_video_is_a_stage_error():
    assert issubclass(NoVideo, StageError) and not issubclass(Refused, StageError)


def test_processors_are_the_ones_this_job_may_use(monkeypatch):
    import os
    # Starwish's job machine reports 8 processors and allows 4.
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert s.processors() == 4


def test_processors_without_an_affinity_call(monkeypatch):
    import os
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert s.processors() == 6
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert s.processors() == 1


def test_speech_is_heard_fast_for_the_judge_and_accurately_for_the_captions():
    assert (s.WINDOW_MODEL, s.WINDOW_BEAM) == ("base", 1)
    assert (s.CAPTION_MODEL, s.CAPTION_BEAM, s.CAPTION_PAD) == ("small", 5, 1.0)


def test_the_speech_settings_can_be_set_from_outside(monkeypatch):
    import importlib
    try:
        for name, value in (("WINDOW_MODEL", "tiny"), ("WINDOW_BEAM", "2"),
                            ("CAPTION_MODEL", "medium"), ("CAPTION_BEAM", "nonsense")):
            monkeypatch.setenv(f"STREAMCLIP_{name}", value)
        importlib.reload(s)
        assert (s.WINDOW_MODEL, s.WINDOW_BEAM, s.CAPTION_MODEL, s.CAPTION_BEAM) == ("tiny", 2, "medium", 5)
        monkeypatch.delenv("STREAMCLIP_WINDOW_MODEL")
        monkeypatch.setenv("STREAMCLIP_WHISPER", "small")           # the older name still works
        importlib.reload(s)
        assert s.WINDOW_MODEL == "small"
    finally:
        monkeypatch.undo()
        importlib.reload(s)
    assert (s.WINDOW_MODEL, s.CAPTION_MODEL) == ("base", "small")
