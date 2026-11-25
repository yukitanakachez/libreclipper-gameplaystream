"""The judge stage: an answer (from a file or from the model) becomes checked picks, and with
no usable answer the job still gets picks, ranked by excitement and marked weak."""

import pytest

from clipper.jobs import read_json, write_json
from streamclip import stages
from streamclip.errors import StageError
from streamclip.judge import call
from test_stream_stages_support import fakes, found_job, good_raw  # noqa: F401  (fixtures)


@pytest.fixture
def raw_file(found_job):
    path = found_job.root / "answer.json"
    write_json(path, good_raw(found_job))
    return path


def answering(raw):
    def judge(root, key, model=call.MODEL, post=None, progress=None):
        return raw
    return judge


def raising(exc):
    def judge(root, key, model=call.MODEL, post=None, progress=None):
        raise exc
    return judge


def test_a_raw_answer_from_a_file_becomes_picks(found_job, raw_file):
    stages.judge(found_job, raw=raw_file)
    data = read_json(found_job.picks)
    assert data["wanted"] == 2 and len(data["picks"]) == 2 and data["picks"][0]["rank"] == 1
    assert all(not p["weak"] for p in data["picks"])
    assert read_json(found_job.judge_raw) == read_json(raw_file)
    status = found_job.get_status()
    assert (status["stage"], status["step"]) == ("make", "ready")


def test_with_a_key_the_model_is_called(found_job, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    asked = []

    def judge(root, key, model=call.MODEL, post=None, progress=None):
        asked.append((root, key))
        return good_raw(found_job)
    monkeypatch.setattr(call, "judge", judge)
    stages.judge(found_job)
    assert asked == [(found_job.root, "k")]
    assert read_json(found_job.picks)["model"] == call.MODEL
    assert read_json(found_job.judge_raw)["candidates"]


def test_the_judges_own_key_comes_first_and_the_served_model_is_recorded(found_job, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "general")
    monkeypatch.setenv("JUDGE_ANTHROPIC_KEY", " judge ")
    keys = []

    def judge(root, key, model=call.MODEL, post=None, progress=None):
        keys.append(key)
        return {**good_raw(found_job), "model": "some-other-model"}
    monkeypatch.setattr(call, "judge", judge)
    stages.judge(found_job)
    assert keys == ["judge"] and read_json(found_job.picks)["model"] == "some-other-model"


def test_no_key_falls_back_and_says_so(found_job):                               # Review Focus 2
    found_job.judge_raw.write_text('{"candidates": []}')       # left by an earlier judging
    lines = []
    stages.judge(found_job, progress=lines.append)
    data = read_json(found_job.picks)
    assert data["model"] == "none" and data["picks"] and all(p["weak"] for p in data["picks"])
    assert "No model key. Ranking by excitement only." in lines
    assert not found_job.judge_raw.exists() and found_job.get_status()["stage"] == "make"


@pytest.mark.parametrize("failure", [RuntimeError("unusable answer"),
                                     call.JudgeError("the reply was not JSON"),
                                     call.KeyRejected("the model service rejected the key (401)")])
def test_a_failed_call_falls_back(found_job, monkeypatch, failure):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(call, "judge", raising(failure))
    lines = []
    stages.judge(found_job, progress=lines.append)
    data = read_json(found_job.picks)
    assert data["model"] == "none" and data["picks"] and all(p["weak"] for p in data["picks"])
    assert any(str(failure) in line for line in lines)


def test_moments_dropped_for_their_content_never_come_back(found_job):
    # Nobody checks a clip before it is posted, so "content:" is final even when nothing else is left.
    rows = [{"n": n, "keep": False, "score": 2, "reason": "content: not for posting",
             "start": 0, "end": 20} for n in read_json(found_job.candidates_index)["ready"]]
    path = found_job.root / "answer.json"
    write_json(path, {"candidates": rows})
    with pytest.raises(StageError, match="ruled out every moment"):
        stages.judge(found_job, raw=path)
    assert not found_job.picks.exists() and read_json(found_job.judge_raw)["candidates"] == rows


def test_a_fallback_after_an_answer_leaves_out_what_it_ruled_out(found_job, monkeypatch):
    from streamclip.judge import checks
    ready = read_json(found_job.candidates_index)["ready"]
    rows = [{"n": ready[0], "keep": False, "score": 2, "reason": "Content: not for posting"}]
    path = found_job.root / "answer.json"
    write_json(path, {"candidates": rows})
    monkeypatch.setattr(checks, "apply_checks", lambda raw, job, wanted: [])
    stages.judge(found_job, raw=path)
    data = read_json(found_job.picks)
    assert data["model"] == "none" and data["picks"]
    assert ready[0] not in [p["candidate"] for p in data["picks"]]
    assert found_job.judge_raw.exists()

def test_moments_the_judge_dropped_fill_the_floor_in_its_own_words(found_job):
    rows = [{"n": n, "keep": False, "score": 3, "reason": "Loud, but there is no story.",
             "start": 0, "end": 20} for n in read_json(found_job.candidates_index)["ready"]]
    path = found_job.root / "answer.json"
    write_json(path, {"model": "the-judge", "candidates": rows})
    stages.judge(found_job, raw=path)
    data = read_json(found_job.picks)
    assert data["model"] == "the-judge" and data["picks"] and all(p["weak"] for p in data["picks"])
    assert data["picks"][0]["reason"] == "Loud, but there is no story."


def test_an_answer_file_that_cannot_be_read_falls_back(found_job):
    path = found_job.root / "answer.json"
    path.write_text("I could not decide")
    lines = []
    stages.judge(found_job, raw=path, progress=lines.append)
    assert read_json(found_job.picks)["model"] == "none"
    assert any("could not be read" in line for line in lines)


def test_judging_before_finding_is_a_plain_failure(tmp_path):
    from streamclip.jobdir import JobDir
    with pytest.raises(StageError, match="Run find first"):
        stages.judge(JobDir(tmp_path / "empty"))


def test_a_slow_judge_keeps_saying_it_is_working(found_job, monkeypatch):
    import time
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def slow(root, key, model=call.MODEL, post=None, progress=None):
        time.sleep(0.5)                 # the real call may think for minutes
        return good_raw(found_job)
    monkeypatch.setattr(call, "judge", slow)
    monkeypatch.setattr(stages, "QUIET_AFTER", 0.1)
    lines = []
    stages.judge(found_job, progress=lines.append)
    assert any(line.startswith("Still working: asking") for line in lines)
