"""The whole thing: one source in, finished clips out, through the stages and the command line."""

import time
from pathlib import Path

import pytest

from clipper import shell
from clipper.jobs import read_json
from streamclip import cli, stages
from streamclip.jobdir import JobDir
from streamclip.judge import call
from test_stream_stages_support import fakes, good_raw  # noqa: F401  (fixtures)

SAMPLE = Path(__file__).resolve().parents[2] / "agents" / "gameplay-clipper" / "tests" / "sample-stream.mp4"


@pytest.fixture
def raw_from_fake_judge(monkeypatch):
    """A judge that answers at once, keeping the two loud moments."""
    monkeypatch.setenv("JUDGE_ANTHROPIC_KEY", "k")
    monkeypatch.setattr(call, "judge",
                        lambda root, key, model=call.MODEL, post=None, progress=None: good_raw(JobDir(root)))


def test_run_goes_from_a_file_to_finished_clips(tmp_path, synthetic_stream, fakes, raw_from_fake_judge):
    job = JobDir(tmp_path / "job")
    lines = []
    stages.run(str(synthetic_stream), job, wanted=2, progress=lines.append)
    listing = read_json(job.out / "clips.json")
    assert len(list(job.out.glob("*.mp4"))) == 2 and [c["rank"] for c in listing["clips"]] == [1, 2]
    assert read_json(job.picks)["model"] == call.MODEL and job.get_status()["stage"] == "done"
    # One voice for all three stages, in order.
    first = [next(i for i, l in enumerate(lines) if l.startswith(start))
             for start in ("Reading the stream", "Asking the judge", "Getting ready to make")]
    assert first == sorted(first)


def test_the_command_line_does_the_same(tmp_path, synthetic_stream, fakes, capsys):
    assert cli.main(["run", str(synthetic_stream), "--job", str(tmp_path / "j"),
                     "--out", str(tmp_path / "o"), "--clips", "2"]) == 0
    assert list((tmp_path / "o").glob("*.mp4")) and (tmp_path / "o" / "clips.json").exists()
    assert "No model key. Ranking by excitement only." in capsys.readouterr().out
    assert cli.main(["status", str(tmp_path / "j")]) == 0
    assert capsys.readouterr().out.startswith("done")


def test_the_three_commands_one_after_another(tmp_path, synthetic_stream, fakes):
    job = str(tmp_path / "j")
    assert cli.main(["find", str(synthetic_stream), "--job", job, "--clips", "2", "--creator", "Some Streamer"]) == 0
    assert cli.main(["judge", job]) == 0
    assert cli.main(["make", job, "--no-sounds"]) == 0
    listing = read_json(tmp_path / "j" / "out" / "clips.json")
    assert listing["clips"] and all(c["weak"] and c["sounds"] == [] for c in listing["clips"])
    assert (tmp_path / "j" / "clips" / "01" / "tag.png").exists()      # the given name made a tag


def test_a_refusal_on_the_command_line_is_recorded(tmp_path, capsys):
    assert cli.main(["run", "https://www.twitch.tv/somestreamer", "--job", str(tmp_path / "j")]) == 2
    status = JobDir(tmp_path / "j").get_status()
    assert status["stage"] == "failed" and status["step"] == "find" and status["error"]


def test_warm_asks_each_lane_for_its_model(monkeypatch):
    called = []
    monkeypatch.setattr("streamclip.signals.ensure_model", lambda: called.append("sound"))
    monkeypatch.setattr("streamclip.windows.ensure_model", lambda: called.append("speech"))
    monkeypatch.setattr("clipper.provision.ensure", lambda items=(): called.append(items))
    monkeypatch.setattr(stages, "_ffmpeg_has_subtitles", lambda: True)
    lines = []
    stages.warm(progress=lines.append)
    assert called == ["sound", "speech", ("model",)] and lines[-1] == "Models are ready."


def test_warm_fetches_an_ffmpeg_when_the_one_in_use_cannot_burn_captions(monkeypatch):
    called = []
    monkeypatch.setattr("streamclip.signals.ensure_model", lambda: None)
    monkeypatch.setattr("streamclip.windows.ensure_model", lambda: None)
    monkeypatch.setattr("clipper.provision.ensure", lambda items=(): called.append(items))
    monkeypatch.setattr(stages, "_ffmpeg_has_subtitles", lambda: False)
    stages.warm(progress=lambda line: None)
    assert called == [("model",), ("ffmpeg",)]


@pytest.mark.live
def test_the_stand_in_sample_end_to_end(tmp_path):
    """Real models, the real judge call, the stand-in sample, two clips. Needs a model key."""
    job = JobDir(tmp_path / "job")
    took = {}
    for name, stage in (("find", lambda: stages.find(str(SAMPLE), job, wanted=2)),
                        ("judge", lambda: stages.judge(job)),
                        ("make", lambda: stages.make(job))):
        started = time.monotonic()
        stage()
        took[name] = time.monotonic() - started
    print("\nseconds per stage: " + ", ".join(f"{name} {seconds:.0f}" for name, seconds in took.items()))
    print(f"clips in {job.out}")
    listing = read_json(job.out / "clips.json")
    assert len(listing["clips"]) >= 1
    for clip in listing["clips"]:
        streams = shell.probe(job.out / clip["file"])["streams"]
        picture = next(s for s in streams if s["codec_type"] == "video")
        assert (picture["width"], picture["height"]) == (1080, 1920)
        assert any(s["codec_type"] == "audio" for s in streams)
