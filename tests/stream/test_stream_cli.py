import subprocess
import sys
import types
from pathlib import Path

import pytest

from streamclip import cli, stages
from streamclip.errors import Refused, StageError
from streamclip.jobdir import JobDir
from streamclip.schema import MakeOptions


def test_find_passes_arguments(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(stages, "find", lambda source, job, wanted, creator, progress: seen.update(
        source=source, root=job.root, wanted=wanted, creator=creator))
    assert cli.main(["find", "https://www.twitch.tv/videos/1", "--job", str(tmp_path), "--clips", "2"]) == 0
    assert seen == {"source": "https://www.twitch.tv/videos/1", "root": tmp_path, "wanted": 2, "creator": ""}


def test_make_options(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(stages, "make", lambda job, out, only, options, progress: seen.update(
        only=only, speed=options.speed, dead_air=options.dead_air, sounds=options.sounds))
    cli.main(["make", str(tmp_path), "--only", "3", "--speed", "1.0", "--dead-air", "0.8", "--no-sounds"])
    assert seen == {"only": 3, "speed": 1.0, "dead_air": 0.8, "sounds": False}


def test_make_defaults(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(stages, "make", lambda job, out, only, options, progress: seen.update(
        root=job.root, job_out=job.out, out=out, only=only, options=options))
    assert cli.main(["make", str(tmp_path)]) == 0
    assert seen == {"root": tmp_path, "job_out": tmp_path / "out", "out": None, "only": None,
                    "options": MakeOptions(1.3, 0.5, True)}
    cli.main(["make", str(tmp_path), "--out", str(tmp_path / "clips")])
    assert seen["out"] == tmp_path / "clips" and seen["job_out"] == tmp_path / "clips"


def test_judge_passes_the_raw_answer(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(stages, "judge", lambda job, raw, progress: seen.update(root=job.root, raw=raw))
    assert cli.main(["judge", str(tmp_path)]) == 0
    assert seen == {"root": tmp_path, "raw": None}
    cli.main(["judge", str(tmp_path), "--raw", str(tmp_path / "picks_raw.json")])
    assert seen["raw"] == tmp_path / "picks_raw.json"


def test_run_passes_arguments(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(stages, "run", lambda source, job, out, wanted, creator, options, progress:
                        seen.update(source=source, root=job.root, out=out, wanted=wanted,
                                    creator=creator, options=options))
    assert cli.main(["run", "stream.mp4", "--job", str(tmp_path), "--out", str(tmp_path / "o"),
                     "--creator", "Speed"]) == 0
    assert seen == {"source": "stream.mp4", "root": tmp_path, "out": tmp_path / "o", "wanted": 6,
                    "creator": "Speed", "options": MakeOptions()}


def test_run_names_its_own_job_folder(tmp_path, monkeypatch):
    roots = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stages, "run", lambda source, job, **kw: roots.append(job.root))
    for source in ("https://www.twitch.tv/videos/2345678901", "https://www.twitch.tv/videos/2345678901",
                   "https://www.twitch.tv/videos/2345678902", "/a/🔥.mp4", "/b/🔥.mp4"):
        cli.main(["run", source])
    assert all(root.parent == Path("stream-jobs") for root in roots)
    assert roots[0] == roots[1] and roots[0].name.startswith("twitch-tv-videos-2345678901")
    assert len({root.name for root in roots}) == 4          # the same source again reuses its folder
    assert all(len(root.name) <= 60 for root in roots)


def test_refused_exits_2_with_the_reason(tmp_path, monkeypatch, capsys):
    def refuse(*a, **k): raise Refused("That broadcast is for subscribers only.")
    monkeypatch.setattr(stages, "find", refuse)
    assert cli.main(["find", "x", "--job", str(tmp_path)]) == 2
    assert "subscribers only" in capsys.readouterr().err
    status = JobDir(tmp_path).get_status()
    assert (status["stage"], status["error"]) == ("failed", "That broadcast is for subscribers only.")


def test_failure_exits_1_and_is_recorded(tmp_path, monkeypatch):
    def fail(*a, **k): raise StageError("The sound could not be fetched.")
    monkeypatch.setattr(stages, "find", fail)
    assert cli.main(["find", "x", "--job", str(tmp_path)]) == 1
    assert JobDir(tmp_path).get_status()["error"] == "The sound could not be fetched."


def test_failure_keeps_where_it_was(tmp_path, monkeypatch, capsys):
    def fail(source, job, **k):
        job.set_status("judge", "asking the model")
        raise RuntimeError("boom")
    monkeypatch.setattr(stages, "run", fail)
    assert cli.main(["run", "x", "--job", str(tmp_path)]) == 1
    status = JobDir(tmp_path).get_status()
    assert (status["stage"], status["step"], status["detail"]) == ("failed", "judge", "asking the model")
    assert capsys.readouterr().err.strip() == "error: boom"
    cli.main(["status", str(tmp_path)])
    assert capsys.readouterr().out.strip() == "failed: boom"


def test_a_failure_without_words_still_says_something(tmp_path, monkeypatch, capsys):
    def fail(*a, **k): raise KeyError()
    monkeypatch.setattr(stages, "judge", fail)
    assert cli.main(["judge", str(tmp_path)]) == 1
    assert JobDir(tmp_path).get_status()["error"] == "KeyError"


def test_the_stages_are_wired(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["find", "x", "--job", str(tmp_path / "job")]) == 2     # no such file: refused
    assert "No file at" in capsys.readouterr().err
    assert JobDir(tmp_path / "job").get_status()["step"] == "find"


@pytest.mark.parametrize("args", (["find", "x", "--job", "J", "--clips", "11"],
                                  ["find", "x", "--job", "J", "--clips", "0"],
                                  ["run", "x", "--clips", "many"],
                                  ["find", "x"], ["make", "J", "--speed", "0"],
                                  ["make", "J", "--only", "0"], ["make", "J", "--dead-air", "-1"], []))
def test_usage_errors(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(args)
    assert not list(tmp_path.iterdir())


def test_clip_count_is_bounded(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["find", "x", "--job", str(tmp_path), "--clips", "11"])


def test_status_prints_one_line(tmp_path, capsys):
    JobDir(tmp_path).set_status("make", "clip 3 of 6")
    cli.main(["status", str(tmp_path)])
    assert capsys.readouterr().out.strip() == "make clip 3 of 6"
    JobDir(tmp_path).set_status("find", "sound", "42%")
    cli.main(["status", str(tmp_path)])
    assert capsys.readouterr().out.strip() == "find sound 42%"


def test_status_of_a_job_that_never_ran(tmp_path, capsys):
    assert cli.main(["status", str(tmp_path / "nope")]) == 0
    assert capsys.readouterr().out.strip() == "new"
    assert not (tmp_path / "nope").exists()


def test_warm(monkeypatch):
    called = []
    monkeypatch.setattr(stages, "warm", lambda progress: called.append(progress))
    assert cli.main(["warm"]) == 0 and len(called) == 1


def test_serve_loads_the_page_only_when_asked(tmp_path, monkeypatch):
    assert "streamclip.web.app" not in sys.modules
    seen = {}
    web = types.ModuleType("streamclip.web")
    app = types.ModuleType("streamclip.web.app")
    app.create_app = lambda jobs_root: ("app", jobs_root)
    monkeypatch.setitem(sys.modules, "streamclip.web", web)
    monkeypatch.setitem(sys.modules, "streamclip.web.app", app)
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(app=app, **kw))
    assert cli.main(["serve", "--port", "9001", "--jobs", str(tmp_path)]) == 0
    assert seen["app"] == ("app", tmp_path) and (seen["host"], seen["port"]) == ("127.0.0.1", 9001)


def test_runs_as_a_module():
    done = subprocess.run([sys.executable, "-m", "streamclip", "--help"], capture_output=True, text=True)
    assert done.returncode == 0
    assert all(name in done.stdout for name in ("find", "judge", "make", "run", "status", "warm", "serve"))


def test_stage_signatures_match_the_plan():
    import inspect

    def names(stage):
        return list(inspect.signature(stage).parameters)
    assert names(stages.find) == ["source", "job", "wanted", "creator", "progress"]
    assert names(stages.judge) == ["job", "raw", "progress"]
    assert names(stages.make) == ["job", "out", "only", "options", "progress"]
    assert names(stages.run) == ["source", "job", "out", "wanted", "creator", "options", "progress"]
    assert names(stages.warm) == ["progress"]
