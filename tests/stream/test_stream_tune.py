"""scripts/stream_tune.py: the measurements Lane N tunes by, on made-up job folders."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stream_tune.py"


@pytest.fixture(scope="module")
def tune():
    spec = importlib.util.spec_from_file_location("stream_tune", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spot(n, peak):
    return {"n": n, "peak": float(peak), "start": float(peak - 60), "end": float(peak + 30),
            "score": 0.5, "fired": []}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _job(root: Path, peaks, clips=None, judged=None, made=None, timings=None, extras=None) -> Path:
    _write(root / "stream.json", {
        "source": "x", "kind": "twitch", "video_id": "1", "title": "t", "creator_login": "c",
        "creator_name": "C", "duration": 7200.0, "started_at": "", "width": 1920, "height": 1080,
        "fps": 60.0, "chapters": [],
        "extras": extras or {"chat": "ok", "viewer_clips": "ok", "avatar": "missing: no picture"}})
    _write(root / "hotspots.json", {"wanted": 6, "hotspots": [_spot(n, p) for n, p in enumerate(peaks, 1)]})
    if clips is not None:
        _write(root / "source" / "viewer_clips.json", clips)
    if judged is not None:
        _write(root / "judge_raw.json", {"candidates": judged})
    if made is not None:
        _write(root / "out" / "clips.json", {"source": "x", "clips": made, "captions": True, "notes": []})
    if timings is not None:
        _write(root / "tuning.json", timings)
    return root


def test_hit_rate_is_the_views_share_of_clips_inside_a_window(tune, tmp_path):
    clips = [{"offset": 990.0, "duration": 20.0, "views": 900, "title": "in"},     # middle 1000: inside 940-1030
             {"offset": 3000.0, "duration": 30.0, "views": 100, "title": "out"}]
    row = tune.measure(_job(tmp_path / "a", [1000, 5000], clips=clips))
    assert row["hit_rate"] == pytest.approx(0.9)


def test_no_viewer_clips_leaves_the_hit_rate_blank(tune, tmp_path):
    row = tune.measure(_job(tmp_path / "a", [1000]))
    assert row["hit_rate"] is None
    assert tune.cell(row["hit_rate"]) == ""


def test_keep_rate_counts_kept_candidates_scored_seven_or_more(tune, tmp_path):
    judged = [{"n": 1, "keep": True, "score": 8}, {"n": 2, "keep": True, "score": 6},
              {"n": 3, "keep": False, "score": 9}, {"n": 4, "keep": True, "score": 7}]
    row = tune.measure(_job(tmp_path / "a", [100, 1000, 2000, 3000], judged=judged))
    assert row["keep_rate"] == pytest.approx(0.5)
    assert tune.measure(_job(tmp_path / "b", [100]))["keep_rate"] is None


def test_spread_is_the_share_of_the_shortlist_in_the_busiest_half_hour(tune, tmp_path):
    row = tune.measure(_job(tmp_path / "a", [100, 900, 1700, 5000]))      # 100..1700 fit in 1800 s
    assert row["spread"] == pytest.approx(0.75)


def test_extras_times_and_dead_air_are_read_from_the_job(tune, tmp_path):
    made = [{"rank": 1, "seconds_cut": 2.5, "cuts": 3}, {"rank": 2, "seconds_cut": 0.0, "cuts": 0}]
    timings = {"stages": {"find": 100.0, "judge": 1.0, "make": 50.0},
               "steps": {"find/sound": 60.0}, "chat_seconds": 12.5}
    row = tune.measure(_job(tmp_path / "a", [100], made=made, timings=timings))
    assert row["extras"] == {"chat": True, "viewer_clips": True, "avatar": False}
    assert row["chat_seconds"] == 12.5
    assert row["stages"] == {"find": 100.0, "judge": 1.0, "make": 50.0}
    assert row["total_seconds"] == pytest.approx(151.0)
    assert row["dead_air"] == [(1, 2.5, 3), (2, 0.0, 0)]


def test_total_weighs_every_viewer_clip_and_candidate_once(tune, tmp_path):
    a = _job(tmp_path / "a", [1000], clips=[{"offset": 990.0, "duration": 20.0, "views": 300, "title": ""}],
             judged=[{"n": 1, "keep": True, "score": 9}])
    b = _job(tmp_path / "b", [1000, 4000], clips=[{"offset": 6000.0, "duration": 20.0, "views": 100, "title": ""}],
             judged=[{"n": 1, "keep": False, "score": 2}, {"n": 2, "keep": False, "score": 3}])
    total = tune.total([tune.measure(a), tune.measure(b)])
    assert total["hit_rate"] == pytest.approx(0.75)
    assert total["keep_rate"] == pytest.approx(1 / 3)


def test_the_table_has_a_row_per_stream_and_a_total(tune, tmp_path, capsys):
    _job(tmp_path / "one", [1000])
    _job(tmp_path / "two", [1000, 2000])
    (tmp_path / "bin").mkdir()                       # not a job: no stream.json
    assert tune.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| one |" in out and "| two |" in out and "| total |" in out and "| bin |" not in out


def test_settings_given_on_the_command_line_reach_the_named_constants(tune):
    from streamclip.signals import curve, loudness
    before = dict(curve.WEIGHTS), loudness.LOUD_FULL_DB
    try:
        tune.apply_settings(["curve.WEIGHTS.voice=0.5", "loudness.LOUD_FULL_DB=9"])
        assert curve.WEIGHTS["voice"] == 0.5 and loudness.LOUD_FULL_DB == 9.0
        with pytest.raises(SystemExit):
            tune.apply_settings(["curve.NOT_THERE=1"])
    finally:
        curve.WEIGHTS.update(before[0])
        loudness.LOUD_FULL_DB = before[1]
