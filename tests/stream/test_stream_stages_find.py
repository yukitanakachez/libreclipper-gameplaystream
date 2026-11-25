"""The find stage, on the two-minute test stream with no model downloaded."""

import time

import pytest

from clipper.jobs import read_json
from streamclip import stages
from streamclip.errors import Refused, StageError
from streamclip.schema import StreamInfo
from test_stream_stages_support import fakes, hears_nothing, job  # noqa: F401  (fixtures)


def test_find_writes_every_file_the_judge_needs(job, synthetic_stream, fakes):
    stages.find(str(synthetic_stream), job, wanted=2)
    assert read_json(job.request) == {"source": str(synthetic_stream), "clips": 2, "creator": ""}
    assert StreamInfo.from_dict(read_json(job.stream)).duration == pytest.approx(120, abs=0.5)
    assert job.audio.exists() and job.signals.exists()
    hot = read_json(job.hotspots)
    assert hot["wanted"] == 2 and 1 <= len(hot["hotspots"]) <= 8
    # The peaks sit where the curve is highest, which is not always on a burst; each burst
    # is inside some hot spot's window.
    for burst in (31, 96):
        assert any(h["start"] <= burst <= h["end"] for h in hot["hotspots"])
    index = read_json(job.candidates_index)
    assert index["ready"] and job.get_status()["stage"] == "judge"
    for n in index["ready"]:
        assert (job.candidate(n) / "transcript.json").exists()
        assert (job.candidate(n) / "window.mp4").exists()
        assert (job.candidate(n) / "sheet.jpg").exists()


def test_a_given_creator_name_fills_the_gap_for_a_file(job, synthetic_stream, fakes):
    stages.find(str(synthetic_stream), job, creator="Some Streamer")
    assert read_json(job.stream)["creator_name"] == "Some Streamer"
    assert read_json(job.request)["creator"] == "Some Streamer"


def test_missing_extras_are_noted_and_nothing_fails(job, synthetic_stream, fakes):
    stages.find(str(synthetic_stream), job)
    extras = read_json(job.stream)["extras"]
    assert all(extras[name].startswith("missing: ") for name in ("chat", "viewer_clips", "avatar"))
    assert not job.chat.exists() and not job.viewer_clips.exists() and not job.avatar.exists()


def test_an_extra_that_blows_up_is_noted_and_nothing_fails(job, synthetic_stream, fakes, monkeypatch):
    real = stages.open_source

    def opened(source):
        src = real(source)
        src.chat_times = lambda: (_ for _ in ()).throw(RuntimeError("chat fell over"))
        src.viewer_clips = lambda: ([], "ok")
        return src
    monkeypatch.setattr(stages, "open_source", opened)
    stages.find(str(synthetic_stream), job)
    extras = read_json(job.stream)["extras"]
    assert extras["chat"] == "missing: chat fell over" and extras["viewer_clips"] == "ok"
    assert read_json(job.viewer_clips) == [] and not job.chat.exists()


def test_a_refusal_passes_straight_through(job):
    with pytest.raises(Refused):
        stages.find("https://www.twitch.tv/somestreamer", job)
    assert not job.audio.exists()
    assert job.get_status()["stage"] == "find"       # so the command line records where it stopped


def test_sound_that_cannot_be_fetched_is_a_plain_failure(job, synthetic_stream, fakes, monkeypatch):
    real = stages.open_source

    def opened(source):
        src = real(source)
        src.fetch_audio = lambda dest: (_ for _ in ()).throw(OSError("disk full"))
        return src
    monkeypatch.setattr(stages, "open_source", opened)
    with pytest.raises(StageError, match="The stream's sound could not be fetched: disk full"):
        stages.find(str(synthetic_stream), job)


def test_a_flat_stream_says_nothing_stood_out(job, synthetic_stream, fakes, monkeypatch):
    monkeypatch.setattr("streamclip.signals.pick_hotspots", lambda signals, wanted: [])
    with pytest.raises(StageError, match="Nothing in this stream stood out enough to clip."):
        stages.find(str(synthetic_stream), job)


def test_progress_is_reported_in_plain_lines(job, synthetic_stream, fakes):
    lines = []
    stages.find(str(synthetic_stream), job, progress=lines.append)
    assert lines[0].startswith("Reading the stream") and any(l.startswith("Windows:") for l in lines)
    assert any(l.startswith("First listen: ") and l.endswith(" s.") for l in lines)


def test_a_slow_step_keeps_saying_it_is_working(job, synthetic_stream, fakes, monkeypatch):
    # Starwish cuts off a step that prints nothing for 300 seconds.
    def slow(pcm):
        time.sleep(0.5)
        return hears_nothing(pcm)
    monkeypatch.setattr("streamclip.signals.events.default_classifier", lambda: slow)
    monkeypatch.setattr(stages, "QUIET_AFTER", 0.1)
    lines = []
    stages.find(str(synthetic_stream), job, progress=lines.append)
    assert any(l.startswith("Still working: signals") for l in lines)


def test_finding_again_clears_what_was_judged_before(job, synthetic_stream, fakes):
    stages.find(str(synthetic_stream), job, wanted=2)
    for stale in (job.picks, job.judge_raw, job.camera):
        stale.write_text("{}")
    stages.find(str(synthetic_stream), job, wanted=2)
    assert not job.picks.exists() and not job.judge_raw.exists() and not job.camera.exists()
