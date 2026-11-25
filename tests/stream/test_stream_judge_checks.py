"""Checking what the judge said: times, lengths, tidy fields, duplicates and ranking."""

import json
from pathlib import Path

import pytest

from clipper import pacing
from streamclip import settings
from streamclip.jobdir import JobDir
from streamclip.judge import checks
from streamclip.judge.checks import apply_checks
from streamclip.schema import Beat, Pick, Transcript

WINDOW1, WINDOW2, WINDOW3 = 540.0, 2000.0, 3000.0
SIX_HOURS = 6 * 3600.0


def spoken(start: float, seconds: float = 90.0, *, punctuated: bool = True) -> list[dict]:
    """A word every half second, 0.4 s long. Every tenth word ends a sentence."""
    words = []
    for k in range(int(seconds * 2)):
        text = f"w{k}." if punctuated and k % 10 == 9 else f"w{k}"
        words.append({"i": k, "text": text, "start": start + 0.5 * k,
                      "end": start + 0.5 * k + 0.4, "prob": 1.0})
    return words


def hotspot(n: int, start: float, score: float = 0.5, length: float = 90.0, peak: float | None = None) -> dict:
    return {"n": n, "peak": start + 60.0 if peak is None else peak, "start": start,
            "end": start + length, "score": score,
            "fired": [{"signal": "voice", "t": start + 60.0, "strength": score, "detail": "Screaming"}]}


def make_job(root: Path, hotspots: list[dict], words: dict[int, list[dict]], *,
             duration: float = SIX_HOURS, title: str = "Ranked until I win a game today okay chat",
             ready: list[int] | None = None) -> JobDir:
    job = JobDir(root)
    (root / "candidates").mkdir(parents=True)
    job.stream.write_text(json.dumps({
        "source": "sample.mp4", "kind": "file", "video_id": "", "title": title,
        "creator_login": "somebody", "creator_name": "Somebody", "duration": duration,
        "started_at": "", "width": 1280, "height": 720, "fps": 30.0, "chapters": [], "extras": {}}))
    job.hotspots.write_text(json.dumps({"wanted": 6, "hotspots": hotspots}))
    numbers = [h["n"] for h in hotspots] if ready is None else ready
    job.candidates_index.write_text(json.dumps(
        {"language": "en", "captions": True, "ready": numbers, "notes": []}))
    for h in hotspots:
        folder = job.candidate(h["n"])
        folder.mkdir()
        mine = words.get(h["n"], [])
        # Lane D may write the window's own length here, so the checks must not lean on it.
        (folder / "transcript.json").write_text(json.dumps({"words": mine, "segments": [], "duration": 90.0}))
    return job


@pytest.fixture
def job(tmp_path) -> JobDir:
    return make_job(tmp_path / "job", [hotspot(1, WINDOW1), hotspot(2, WINDOW2)],
                    {1: spoken(WINDOW1), 2: spoken(WINDOW2)})


@pytest.fixture
def job_with_silent_candidate(tmp_path) -> JobDir:
    return make_job(tmp_path / "job", [hotspot(1, WINDOW1), hotspot(3, WINDOW3)], {1: spoken(WINDOW1)})


@pytest.fixture
def job_with_overlapping_windows(tmp_path) -> JobDir:
    return make_job(tmp_path / "job", [hotspot(1, 540.0), hotspot(2, 585.0)],
                    {1: spoken(540.0), 2: spoken(585.0)})


def verdict(n=1, *, keep=True, start=30.0, end=65.0, score=8, beats=(), protect=(), **more) -> dict:
    return {"n": n, "keep": keep, "score": score, "reason": "Build-up, then the scare lands.",
            "start": start, "end": end, "title": "He walked right into it",
            "caption": "wait for the scream", "hashtags": ["#gaming"],
            "beats": [{"kind": kind, "t": t} for kind, t in beats],
            "protect": [{"start": a, "end": b} for a, b in protect]} | more


def raw(n=1, **fields) -> dict:
    return {"model": "claude-opus-5-5", "candidates": [verdict(n, **fields)]}


def also(*answers: dict) -> dict:
    return {"candidates": [c for answer in answers for c in answer["candidates"]]}


def many_raw(scores: list[int]) -> dict:
    """Candidate 1 answered once per score, each a different stretch of its window."""
    return {"model": "claude-opus-5-5", "candidates": [
        verdict(n, score=score) for n, score in enumerate(scores, start=1)]}


def estimated_finished(pick: Pick, job: JobDir) -> float:
    transcript = Transcript.from_dict(json.loads((job.candidate(pick.candidate) / "transcript.json").read_text()))
    words = [w for w in transcript.words if w.end > pick.start and w.start < pick.end]
    pace = pacing.Pacing(True, settings.DEAD_AIR, settings.SPEED)
    return pacing.finished_duration(pacing.keep_intervals(words, pick.start, pick.end, pace), pace)


def test_window_seconds_become_stream_seconds(job):
    pick = apply_checks(raw(n=1, start=30.0, end=65.0, beats=[("payoff", 61.0)], protect=[(40.0, 43.0)]), job, 6)[0]
    assert 569.5 <= pick.start <= 570.5 and 604.5 <= pick.end <= 605.6        # snapped to word edges
    assert pick.beats == [Beat("payoff", 601.0)] and pick.protect == [(580.0, 583.0)]
    assert pick.candidate == 1 and pick.rank == 1 and pick.weak is False


def test_a_pick_survives_a_round_trip_through_json(job):
    pick = apply_checks(raw(n=1, beats=[("payoff", 61.0)], protect=[(40.0, 43.0)]), job, 6)[0]
    assert Pick.from_dict(json.loads(json.dumps(pick.to_dict()))) == pick
    assert isinstance(pick.score, int)


def test_times_outside_the_window_are_pulled_in(job):
    pick = apply_checks(raw(n=1, start=-5.0, end=40.0), job, 6)[0]
    assert pick.start >= 540.0
    pick = apply_checks(raw(n=1, start=50.0, end=400.0), job, 6)[0]
    assert pick.end <= 630.0


def test_too_long_keeps_the_ending(job):
    pick = apply_checks(raw(n=1, start=5.0, end=85.0), job, 6)[0]
    assert pick.end >= 624.0 and (pick.end - pick.start) / 1.3 <= 40.0
    assert (pick.start - WINDOW1 + 0.1) % 5.0 < 0.2                 # it opens on a sentence


def test_too_short_reaches_back_for_more_build_up(job):
    pick = apply_checks(raw(n=1, start=60.0, end=66.0), job, 6)[0]
    assert pick.end <= 606.6 and estimated_finished(pick, job) >= 15.0
    assert pick.end - pick.start < 30.0                             # and no further than it needs


def test_too_short_with_nothing_to_reach_back_to_is_dropped(job):
    assert [p for p in apply_checks(raw(n=1, start=0.0, end=8.0), job, 6) if not p.weak] == []


def test_a_window_with_no_words_is_kept_unsnapped(job_with_silent_candidate):   # Review Focus 1
    pick = apply_checks(raw(n=3, start=30.0, end=60.0), job_with_silent_candidate, 6)[0]
    assert (pick.start, pick.end) == (WINDOW3 + 30.0, WINDOW3 + 60.0)


def test_a_window_with_no_words_still_gets_its_length_fixed(job_with_silent_candidate):
    long = apply_checks(raw(n=3, start=10.0, end=90.0), job_with_silent_candidate, 6)[0]
    assert long.end == WINDOW3 + 90.0 and long.end - long.start == pytest.approx(52.0)
    short = apply_checks(raw(n=3, start=50.0, end=60.0), job_with_silent_candidate, 6)[0]
    assert short.end == WINDOW3 + 60.0 and (short.end - short.start) / 1.3 == pytest.approx(15.0, abs=0.1)


def test_words_without_punctuation_still_snap(tmp_path):
    job = make_job(tmp_path / "job", [hotspot(1, WINDOW1)], {1: spoken(WINDOW1, punctuated=False)})
    pick = apply_checks(raw(n=1, start=5.2, end=85.2), job, 6)[0]
    assert pick.end >= 624.0 and 15.0 <= estimated_finished(pick, job) <= 40.0


def test_a_protected_quiet_stretch_counts_towards_the_length(tmp_path):
    words = [w for w in spoken(WINDOW1) if not WINDOW1 + 50.0 <= w["start"] < WINDOW1 + 62.0]
    job = make_job(tmp_path / "job", [hotspot(1, WINDOW1)], {1: words})
    pick = apply_checks(raw(n=1, start=45.0, end=65.0, protect=[(50.0, 62.0)]), job, 6)[0]
    assert pick.start >= WINDOW1 + 44.0                    # 20 s of source is already enough


def test_the_same_moment_twice_keeps_the_better_one(job_with_overlapping_windows):
    answer = also(raw(n=1, start=50.0, end=85.0, score=6), raw(n=2, start=5.0, end=40.0, score=9))
    picks = apply_checks(answer, job_with_overlapping_windows, 6)
    assert len(picks) == 1 and picks[0].score == 9


def test_ranked_best_first_and_cut_to_the_number_wanted(tmp_path):
    job = make_job(tmp_path / "job", [hotspot(n, 1000.0 * n) for n in (1, 2, 3)],
                   {n: spoken(1000.0 * n) for n in (1, 2, 3)})
    picks = apply_checks(many_raw(scores=[5, 9, 7]), job, 2)
    assert [(p.rank, p.score) for p in picks] == [(1, 9), (2, 7)]
    tied = apply_checks(many_raw(scores=[7, 7, 7]), job, 6)
    assert [p.candidate for p in tied] == [1, 2, 3]                 # ties go to the earlier moment


def test_dropped_candidates_and_unknown_numbers_are_left_out(job):
    picks = apply_checks(also(raw(n=1, keep=False), raw(n=99)), job, 6)
    assert [p for p in picks if not p.weak] == [] and 99 not in [p.candidate for p in picks]


def test_a_candidate_that_is_not_ready_is_left_out(tmp_path):
    job = make_job(tmp_path / "job", [hotspot(1, WINDOW1), hotspot(2, WINDOW2)],
                   {1: spoken(WINDOW1), 2: spoken(WINDOW2)}, ready=[1])
    assert [p.candidate for p in apply_checks(also(raw(n=1), raw(n=2, score=10)), job, 6)] == [1]


def test_fields_are_tidied(job):
    pick = apply_checks(raw(n=1, title="one two three four five six seven eight nine ten",
                            hashtags=["gaming", "#Funny", "#funny"], score=14,
                            beats=[("payoff", 61.0), ("dance", 62.0), ("build_up", 60.0), ("reaction", 500.0)]), job, 6)[0]
    assert len(pick.title.split()) == 8 and pick.hashtags == ["#gaming", "#Funny"] and pick.score == 10
    assert [b.kind for b in pick.beats] == ["payoff"]          # unknown kinds and times outside the pick go


def test_an_empty_title_falls_back_to_the_streams(job):
    pick = apply_checks(raw(n=1, title="  "), job, 6)[0]
    assert pick.title == "Ranked until I win a game today okay"


@pytest.mark.parametrize("answer", [
    {}, {"candidates": "none"}, {"candidates": [None, 7, {"n": "1"}, {"n": 1}]},
    {"candidates": [verdict(1, start="soon", end=None)]},
    {"candidates": [verdict(1, start=float("nan"), end=60.0)]},
    {"candidates": [verdict(1, start=60.0, end=30.0)]}])
def test_a_broken_answer_gives_no_picks_and_no_crash(job, answer):
    assert [p for p in apply_checks(answer, job, 6) if not p.weak] == []


def test_broken_fields_do_not_cost_a_good_pick(job):
    row = verdict(1) | {"score": "great", "hashtags": None, "beats": [{"kind": "payoff", "t": "x"}, 4],
                        "protect": "none", "caption": None, "reason": None}
    pick = apply_checks({"candidates": [row]}, job, 6)[0]
    assert (pick.score, pick.hashtags, pick.beats, pick.protect, pick.caption) == (1, [], [], [], "")


def test_the_judge_is_not_asked_for_build_up_beats():
    # There is no rising sound to put on one, so the kind is gone from what the judge may answer.
    from pathlib import Path
    folder = Path(checks.__file__).parent
    assert "build_up" not in (folder / "answer_schema.json").read_text()
    assert "build_up" not in (folder / "prompt.md").read_text()
    assert "build_up" not in checks.BEAT_KINDS
