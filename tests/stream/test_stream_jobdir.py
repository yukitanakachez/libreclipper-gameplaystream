from streamclip.jobdir import JobDir, nn, slug


def test_paths(tmp_path):
    job = JobDir(tmp_path)
    assert job.stream == tmp_path / "stream.json"
    assert job.audio == tmp_path / "source" / "audio.m4a"
    assert job.candidate(7) == tmp_path / "candidates" / "07"
    assert job.clip(1) == tmp_path / "clips" / "01"
    assert job.out == tmp_path / "out" and job.picks == tmp_path / "picks.json"


def test_every_path_in_the_plan(tmp_path):
    job = JobDir(tmp_path)
    assert {name: str(getattr(job, name).relative_to(tmp_path)) for name in (
        "request", "status", "stream", "audio", "chat", "viewer_clips", "avatar", "signals",
        "hotspots", "candidates_index", "judge_raw", "picks", "camera", "out")} == {
        "request": "request.json", "status": "status.json", "stream": "stream.json",
        "audio": "source/audio.m4a", "chat": "source/chat.json",
        "viewer_clips": "source/viewer_clips.json", "avatar": "source/avatar.jpg",
        "signals": "signals.json", "hotspots": "hotspots.json",
        "candidates_index": "candidates/index.json", "judge_raw": "judge_raw.json",
        "picks": "picks.json", "camera": "camera.json", "out": "out"}


def test_out_can_live_elsewhere(tmp_path):
    assert JobDir(tmp_path / "job", out=tmp_path / "clips").out == tmp_path / "clips"
    assert JobDir(str(tmp_path)).root == tmp_path


def test_status_roundtrip(tmp_path):
    job = JobDir(tmp_path)
    job.set_status("find", "sound", "42%")
    assert job.get_status()["stage"] == "find" and job.get_status()["error"] is None
    assert set(job.get_status()) == {"stage", "step", "detail", "error", "updated"}


def test_status_before_anything_ran(tmp_path):
    job = JobDir(tmp_path / "nothing-yet")
    assert job.get_status() == {"stage": "new", "step": "", "detail": "", "error": None, "updated": ""}
    assert not job.root.exists()


def test_nn_and_slug():
    assert nn(7) == "07"
    assert slug("He REALLY did that?!") == "he-really-did-that"
    assert slug("🔥🔥") == "clip" and slug("日本語") == "clip"        # Review Focus 4
    assert len(slug("word " * 40)) <= 50


def test_slug_is_safe_as_a_file_name():
    assert slug("../../etc/passwd") == "etc-passwd"
    assert slug("Pokémon 🔥 run #3") == "pokemon-run-3"
    assert not slug("a" * 49 + " b").endswith("-")
