"""The test page's API, with the three stages faked: they write files into the job folder."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipper.jobs import read_json, write_json
from streamclip import stages
from streamclip.errors import Refused
from streamclip.jobdir import JobDir, nn
from streamclip.schema import (Beat, ClipResult, Fired, Hotspot, Pick, Signals, StreamInfo)


def create_app(jobs_root):
    """Imported when called, never at collection: Lane A's serve test checks the page is loaded
    only when asked, so no test module may import it up front."""
    from streamclip.web.app import create_app as make
    return make(jobs_root)


FAST = 3.0          # seconds a test waits for the worker before giving up


def write_finished_job(job: JobDir, wanted: int = 2, seconds: int = 600, peak_value: float = 0.97,
                       video: bytes = b"\x00\x00\x00\x18ftypmp42") -> None:
    """Every file a finished job has, small and made up."""
    info = StreamInfo(source="https://www.twitch.tv/videos/1", kind="twitch", video_id="1",
                      title="Late night ranked grind", creator_login="someone",
                      creator_name="Someone", duration=float(seconds), started_at="2026-11-20T02:00:00Z",
                      width=1280, height=720, fps=30.0, chapters=[],
                      extras={"chat": "ok", "viewer_clips": "missing: none were found", "avatar": "ok"})
    write_json(job.stream, info.to_dict())
    curve = [0.1 + 0.3 * ((i * 7) % 11) / 11 for i in range(seconds)]
    peaks = [seconds * f for f in (0.2, 0.45, 0.7, 0.9)]
    for p in peaks:
        curve[int(p)] = 0.8
    curve[int(peaks[1])] = peak_value
    zeros = [0.0] * seconds
    write_json(job.signals, Signals(duration=float(seconds), loudness=curve, surge=zeros, voice=zeros,
                                    chat=None, clips=None, penalty=zeros, excitement=curve).to_dict())
    hotspots = [Hotspot(n=i + 1, peak=p, start=max(0.0, p - 60), end=min(seconds, p + 30),
                        score=curve[int(p)],
                        fired=[Fired("voice", p, 0.9, "shouting at %d s" % p),
                               Fired("chat", p + 2, 0.6, "chat x4")])
                for i, p in enumerate(peaks)]
    write_json(job.hotspots, {"wanted": wanted, "hotspots": [h.to_dict() for h in hotspots]})
    for h in hotspots:
        (job.candidate(h.n)).mkdir(parents=True, exist_ok=True)
        (job.candidate(h.n) / "sheet.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    keep = {1: True, 2: True, 3: False, 4: True}
    write_json(job.judge_raw, {"model": "fake", "candidates": [
        {"n": h.n, "keep": keep[h.n], "score": 9 - h.n, "reason": f"Reason for moment {h.n}.",
         "start": 5.0, "end": 40.0, "title": f"Moment {h.n}", "caption": "c", "hashtags": [],
         "beats": [], "protect": []} for h in hotspots]})
    picks = [Pick(rank=r, candidate=n, start=hotspots[n - 1].start + 5, end=hotspots[n - 1].start + 40,
                  score=9 - n, reason=f"Reason for moment {n}.", title=f"Moment {n}", caption="c",
                  hashtags=["#gaming"], beats=[Beat("payoff", hotspots[n - 1].peak)], protect=[],
                  weak=False)
             for r, n in enumerate([2, 1, 4][:wanted], start=1)]
    write_json(job.picks, {"model": "fake", "wanted": wanted, "picks": [p.to_dict() for p in picks]})
    clips = []
    for p in picks:
        name = f"{nn(p.rank)}-clip"
        job.out.mkdir(parents=True, exist_ok=True)
        (job.out / f"{name}.mp4").write_bytes(video)
        (job.out / f"{name}.txt").write_text("c\n\n#gaming\n")
        clips.append(ClipResult(rank=p.rank, file=f"{name}.mp4", caption_file=f"{name}.txt",
                                title=p.title, score=p.score, reason=p.reason,
                                source_start=p.start, source_end=p.end, duration=27.0, weak=False,
                                layout=["split"], cuts=3, seconds_cut=4.2, sounds=["boom"],
                                captions=True).to_dict())
    write_json(job.out / "clips.json", {"clips": clips, "notes": []})
    job.set_status("done", "make", f"{len(clips)} of {len(clips)}")


def fake_run(source, job, out=None, wanted=6, creator="", options=None, progress=print):
    job.set_status("find", "describe")
    progress("describing the stream")
    write_finished_job(job, wanted=min(wanted, 3))


def fake_make(job, out=None, only=None, options=None, progress=print):
    job.set_status("done", "make", "1 of 1")


def refuse_with(reason):
    def run(*_args, **_kwargs):
        raise Refused(reason)
    return run


def wait_until_done(client, job_id):
    deadline = time.monotonic() + FAST
    while time.monotonic() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"]["stage"] in ("done", "failed"):
            return
        time.sleep(0.02)
    raise AssertionError("the job did not finish")


def wait_until_idle(client):
    assert client.app.state.runner.wait_idle(FAST), "the worker did not go idle"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "run", fake_run)
    monkeypatch.setattr(stages, "make", fake_make)
    return TestClient(create_app(tmp_path))


@pytest.fixture
def finished_job(tmp_path):
    write_finished_job(JobDir(tmp_path / "done-job-20261123-120000"))
    write_json(tmp_path / "other-job" / "stream.json", {"secret": True})
    (tmp_path / "secret").write_text("not for the page")
    return "done-job-20261123-120000"


@pytest.fixture
def finished_job_with_long_curve(tmp_path):
    write_finished_job(JobDir(tmp_path / "long-20261123-120000"), seconds=6 * 3600 + 123)
    return "long-20261123-120000"


@pytest.fixture
def seen(monkeypatch):
    got = {}

    def make(job, out=None, only=None, options=None, progress=print):
        got.update(only=only, speed=options.speed, dead_air=options.dead_air, sounds=options.sounds)
    monkeypatch.setattr(stages, "make", make)
    return got


@pytest.fixture
def slow_run(monkeypatch):
    release = threading.Event()

    def run(source, job, **_kwargs):
        job.set_status("find", "describe")
        release.wait(FAST)
    monkeypatch.setattr(stages, "run", run)
    yield
    release.set()


def test_a_job_runs_and_reports(client):
    job = client.post("/api/jobs", json={"source": "https://www.twitch.tv/videos/1", "clips": 2,
                                         "creator": ""}).json()["id"]
    wait_until_done(client, job)
    body = client.get(f"/api/jobs/{job}").json()
    assert body["status"]["stage"] == "done" and body["stream"]["title"]
    assert len(body["clips"]["clips"]) == 2 and body["judged"][0]["reason"]


def test_a_job_reports_nulls_before_its_files_exist(client, slow_run):
    job = client.post("/api/jobs", json={"source": "a", "clips": 6, "creator": ""}).json()["id"]
    body = client.get(f"/api/jobs/{job}").json()
    assert body["status"]["stage"] in ("queued", "find")
    assert all(body[key] is None for key in ("stream", "hotspots", "judged", "picks", "clips"))


def test_ids_are_safe_names(client):
    job = client.post("/api/jobs", json={"source": "../../etc/passwd", "clips": 6, "creator": ""}).json()["id"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,80}", job)


def test_two_jobs_for_one_source_in_one_second_get_two_folders(client, slow_run):
    a = client.post("/api/jobs", json={"source": "same", "clips": 6, "creator": ""}).json()["id"]
    b = client.post("/api/jobs", json={"source": "same", "clips": 6, "creator": ""}).json()["id"]
    assert a != b


def test_a_refusal_is_shown_as_the_jobs_error(client, monkeypatch):
    monkeypatch.setattr(stages, "run", refuse_with("That broadcast is for subscribers only."))
    job = client.post("/api/jobs", json={"source": "x", "clips": 6, "creator": ""}).json()["id"]
    wait_until_done(client, job)
    status = client.get(f"/api/jobs/{job}").json()["status"]
    assert status["stage"] == "failed" and status["error"] == "That broadcast is for subscribers only."


def test_any_failure_keeps_the_stage_it_happened_in(client, monkeypatch):
    def broken(source, job, **_kwargs):
        job.set_status("judge", "asking the model")
        raise RuntimeError("the model said no")
    monkeypatch.setattr(stages, "run", broken)
    job = client.post("/api/jobs", json={"source": "x", "clips": 6, "creator": ""}).json()["id"]
    wait_until_done(client, job)
    status = client.get(f"/api/jobs/{job}").json()["status"]
    assert (status["stage"], status["step"], status["error"]) == ("failed", "judge", "the model said no")


def test_clip_count_is_checked(client):
    assert client.post("/api/jobs", json={"source": "x", "clips": 11, "creator": ""}).status_code == 422
    assert client.post("/api/jobs", json={"source": "x", "clips": 0, "creator": ""}).status_code == 422
    assert client.post("/api/jobs", json={"source": " ", "clips": 6, "creator": ""}).status_code == 422


def test_the_request_is_passed_to_the_stages(client, monkeypatch):
    got = {}

    def run(source, job, out=None, wanted=6, creator="", options=None, progress=print):
        got.update(source=source, wanted=wanted, creator=creator, root=job.root)
    monkeypatch.setattr(stages, "run", run)
    job = client.post("/api/jobs", json={"source": " x.mp4 ", "clips": 3, "creator": "Kai"}).json()["id"]
    wait_until_idle(client)
    assert got["source"] == "x.mp4" and got["wanted"] == 3 and got["creator"] == "Kai"
    assert read_json(got["root"] / "request.json") == {"source": "x.mp4", "clips": 3, "creator": "Kai"}
    assert got["root"].name == job


def test_jobs_are_listed_newest_first(client):
    first = client.post("/api/jobs", json={"source": "first", "clips": 2, "creator": ""}).json()["id"]
    wait_until_idle(client)
    time.sleep(0.02)
    second = client.post("/api/jobs", json={"source": "second", "clips": 2, "creator": ""}).json()["id"]
    wait_until_idle(client)
    listed = client.get("/api/jobs").json()
    assert [j["id"] for j in listed] == [second, first]
    assert listed[0]["source"] == "second" and listed[0]["status"]["stage"] == "done"


def test_judged_joins_hotspots_answer_and_picks(client, finished_job):
    rows = {r["n"]: r for r in client.get(f"/api/jobs/{finished_job}").json()["judged"]}
    assert rows[2]["keep"] is True and rows[2]["rank"] == 1 and rows[2]["score"] == 7
    assert rows[3]["keep"] is False and rows[3]["rank"] is None and rows[3]["reason"]
    assert rows[1]["fired"][0]["signal"] == "voice" and rows[1]["peak"] > 0
    assert rows[1]["sheet"] == "candidates/01/sheet.jpg"
    assert client.get(f"/api/jobs/{finished_job}/files/{rows[1]['sheet']}").status_code == 200


def test_an_unknown_job_is_not_found(client):
    assert client.get("/api/jobs/nope-20261123-000000").status_code == 404
    assert client.get("/api/jobs/nope-20261123-000000/curve").status_code == 404


def test_curve_is_downsampled(client, finished_job_with_long_curve):
    body = client.get(f"/api/jobs/{finished_job_with_long_curve}/curve").json()
    points = body["points"]
    assert len(points) <= 1200 and max(points) == pytest.approx(0.97)  # buckets keep their highest
    assert body["duration"] == pytest.approx(6 * 3600 + 123)


def test_a_short_curve_is_sent_whole(client, finished_job):
    points = client.get(f"/api/jobs/{finished_job}/curve").json()["points"]
    assert len(points) == 600


def test_files_are_served_and_escapes_are_refused(client, finished_job):
    assert client.get(f"/api/jobs/{finished_job}/files/out/01-clip.mp4").status_code == 200
    assert client.get(f"/api/jobs/{finished_job}/files/../other-job/stream.json").status_code == 404
    assert client.get(f"/api/jobs/{finished_job}/files/%2e%2e/%2e%2e/secret").status_code == 404
    assert client.get(f"/api/jobs/{finished_job}/files/%2e%2e/other-job/stream.json").status_code == 404
    assert client.get("/api/jobs/..%2f..%2fetc/files/passwd").status_code in (400, 404)
    assert client.get(f"/api/jobs/{finished_job}/files/out").status_code == 404


def test_upload_then_run(client):
    path = client.post("/api/upload?name=my stream.mp4", content=b"\x00" * 1024).json()["path"]
    assert Path(path).exists() and Path(path).parent.name == "uploads" and "/" not in Path(path).name
    assert Path(path).suffix == ".mp4" and Path(path).stat().st_size == 1024
    job = client.post("/api/jobs", json={"source": path, "clips": 2, "creator": ""}).json()["id"]
    wait_until_done(client, job)
    assert client.get(f"/api/jobs/{job}").json()["status"]["stage"] == "done"


def test_an_upload_never_leaves_the_uploads_folder(client, tmp_path):
    path = Path(client.post("/api/upload?name=../../evil.mp4", content=b"x").json()["path"])
    assert path.parent == (tmp_path / "uploads").resolve()
    again = Path(client.post("/api/upload?name=../../evil.mp4", content=b"y").json()["path"])
    assert again != path and path.read_bytes() == b"x"


def test_an_empty_upload_is_refused(client):
    assert client.post("/api/upload?name=a.mp4", content=b"").status_code == 400


def test_uploads_are_not_listed_as_jobs(client):
    client.post("/api/upload?name=a.mp4", content=b"x")
    assert client.get("/api/jobs").json() == []


def test_remake_passes_the_knobs(client, finished_job, seen):
    assert client.post(f"/api/jobs/{finished_job}/clips/2/remake",
                       json={"speed": 1.0, "dead_air": 0.8, "sounds": False}).json() == {"ok": True}
    wait_until_idle(client)
    assert seen == {"only": 2, "speed": 1.0, "dead_air": 0.8, "sounds": False}


def test_remake_checks_its_clip_and_knobs(client, finished_job):
    knobs = {"speed": 1.3, "dead_air": 0.5, "sounds": True}
    assert client.post(f"/api/jobs/{finished_job}/clips/9/remake", json=knobs).status_code == 404
    assert client.post(f"/api/jobs/{finished_job}/clips/1/remake",
                       json={**knobs, "speed": 9}).status_code == 422
    assert client.post("/api/jobs/nope-1/clips/1/remake", json=knobs).status_code == 404


def test_a_failed_remake_is_shown(client, finished_job, monkeypatch):
    def broken(job, **_kwargs):
        job.set_status("make", "rendering", "clip 1")
        raise RuntimeError("ffmpeg fell over")
    monkeypatch.setattr(stages, "make", broken)
    client.post(f"/api/jobs/{finished_job}/clips/1/remake", json={"speed": 1.3, "dead_air": 0.5, "sounds": True})
    wait_until_idle(client)
    status = client.get(f"/api/jobs/{finished_job}").json()["status"]
    assert status["stage"] == "failed" and status["error"] == "ffmpeg fell over"


def test_one_job_at_a_time(client, slow_run):
    a = client.post("/api/jobs", json={"source": "a", "clips": 6, "creator": ""}).json()["id"]
    b = client.post("/api/jobs", json={"source": "b", "clips": 6, "creator": ""}).json()["id"]
    assert client.get(f"/api/jobs/{b}").json()["status"]["stage"] == "queued"
    assert a != b


def test_a_job_left_running_by_a_stopped_page_is_marked_failed(tmp_path):
    JobDir(tmp_path / "old-20261122-100000").set_status("make", "rendering", "2 of 6")
    write_json(tmp_path / "older-20261122-090000" / "status.json",
               {"stage": "queued", "step": "", "detail": "", "error": None, "updated": ""})
    client = TestClient(create_app(tmp_path))
    for job in ("old-20261122-100000", "older-20261122-090000"):
        status = client.get(f"/api/jobs/{job}").json()["status"]
        assert status["stage"] == "failed" and "stopped" in status["error"]
