"""The clip floor: weak filler when the judge keeps too few, and picks with no judge at all."""

import pytest

from streamclip.judge.checks import CONTENT, apply_checks, fallback_picks, floor_for
from test_stream_judge_checks import SIX_HOURS, hotspot, make_job, spoken, verdict

EXCITEMENT = {1: 0.40, 2: 0.55, 3: 0.95, 4: 0.30, 5: 0.80, 6: 0.65, 7: 0.20, 8: 0.70}


@pytest.fixture
def six_hour_job(tmp_path):
    """Eight ready candidates spread over six hours. Candidate 3 is the most exciting."""
    spots = [hotspot(n, 2000.0 * n, score=EXCITEMENT[n]) for n in EXCITEMENT]
    return make_job(tmp_path / "job", spots, {n: spoken(2000.0 * n) for n in EXCITEMENT},
                    duration=SIX_HOURS)


@pytest.fixture
def three_minute_job(tmp_path):
    """A stream shorter than two windows: both clamp to it, and the first peak is at 31 s."""
    spots = [hotspot(1, 0.0, score=0.9, length=61.0, peak=31.0),
             hotspot(2, 36.0, score=0.8, length=90.0, peak=96.0)]
    return make_job(tmp_path / "job", spots, {1: spoken(0.0, 61.0), 2: spoken(36.0, 90.0)},
                    duration=180.0)


def raw_with(kept: int, dropped: int, dropped_scores=(3, 6, 2, 5, 4, 1)) -> dict:
    rows = [verdict(n, score=10 - n) for n in range(1, kept + 1)]
    rows += [verdict(kept + k + 1, keep=False, score=dropped_scores[k], reason="Loud, but nothing happens.")
             for k in range(dropped)]
    return {"model": "claude-opus-5-5", "candidates": rows}


def hottest_candidate(job) -> int:
    return max(EXCITEMENT, key=EXCITEMENT.get)


@pytest.mark.parametrize("minutes, wanted, floor", [
    (360, 6, 5), (45, 6, 5), (44, 6, 5), (40, 6, 5), (24, 6, 3), (10, 6, 1), (3, 6, 1), (360, 2, 2), (360, 10, 5)])
def test_floor(minutes, wanted, floor):
    assert floor_for(minutes * 60, wanted) == floor


def test_dropped_candidates_fill_up_to_the_floor_marked_weak(six_hour_job):
    picks = apply_checks(raw_with(kept=2, dropped=6), six_hour_job, 6)
    assert len(picks) == 5 and [p.weak for p in picks] == [False, False, True, True, True]
    assert picks[2].score >= picks[3].score >= picks[4].score
    assert [p.score for p in picks[2:]] == [6, 5, 4] and [p.rank for p in picks] == [1, 2, 3, 4, 5]
    assert picks[2].reason == "Loud, but nothing happens." and picks[2].title


def test_enough_kept_candidates_get_no_filler(six_hour_job):
    picks = apply_checks(raw_with(kept=5, dropped=3), six_hour_job, 6)
    assert len(picks) == 5 and not any(p.weak for p in picks)


def test_filler_stops_at_the_floor_not_at_the_number_wanted(six_hour_job):
    assert len(apply_checks(raw_with(kept=0, dropped=6), six_hour_job, 10)) == 5
    assert len(apply_checks(raw_with(kept=1, dropped=6), six_hour_job, 2)) == 2


def test_candidates_dropped_for_content_never_come_back(six_hour_job):
    answer = raw_with(kept=2, dropped=6)
    best_dropped = max((c for c in answer["candidates"] if not c["keep"]), key=lambda c: c["score"])
    best_dropped["reason"] = " Content: the punchline is a slur."
    picks = apply_checks(answer, six_hour_job, 6)
    assert len(picks) == 5 and best_dropped["n"] not in [p.candidate for p in picks]
    assert [p.score for p in picks[2:]] == [5, 4, 3] and CONTENT == "content:"
    everything = {"candidates": [verdict(n, keep=False, reason="content: someone is really hurt")
                                 for n in EXCITEMENT]}
    assert apply_checks(everything, six_hour_job, 6) == []


def test_a_dropped_candidate_with_unusable_times_is_cut_around_its_peak(six_hour_job):
    answer = raw_with(kept=4, dropped=1)
    answer["candidates"][-1] |= {"start": 0.0, "end": 0.0}
    filler = apply_checks(answer, six_hour_job, 6)[-1]
    assert filler.weak and filler.candidate == 5
    assert filler.start < 2000.0 * 5 + 60.0 < filler.end


def test_candidates_the_judge_skipped_fill_last_by_excitement(six_hour_job):
    picks = apply_checks(raw_with(kept=2, dropped=1), six_hour_job, 6)
    assert [(p.candidate, p.weak) for p in picks] == [(1, False), (2, False), (3, True), (5, True), (8, True)]
    assert picks[3].score == 1 and "did not rate" in picks[3].reason


def test_filler_never_repeats_a_moment_already_picked(tmp_path):
    job = make_job(tmp_path / "job", [hotspot(1, 540.0), hotspot(2, 585.0)],
                   {1: spoken(540.0), 2: spoken(585.0)})
    answer = {"candidates": [verdict(1, start=50.0, end=85.0),
                             verdict(2, keep=False, start=5.0, end=40.0, score=7)]}
    assert [p.candidate for p in apply_checks(answer, job, 6)] == [1]


def test_fallback_ranks_by_excitement_and_marks_everything_weak(six_hour_job):       # Review Focus 2
    picks = fallback_picks(six_hour_job, 6)
    assert len(picks) == 5 and all(p.weak for p in picks)          # weak clips only ever fill the floor
    assert picks[0].candidate == hottest_candidate(six_hour_job)
    assert all(20.0 <= p.end - p.start <= 52.0 for p in picks)
    assert [p.candidate for p in picks] == [3, 5, 8, 6, 2] and [p.rank for p in picks] == [1, 2, 3, 4, 5]
    first = picks[0]
    assert first.caption == "" and first.hashtags == []
    assert first.start < 6060.0 < first.end and isinstance(first.score, int) and 1 <= first.score <= 10
    assert "Screaming" in first.reason


def test_a_fallback_pick_has_no_title_of_its_own(six_hour_job):
    # No judge means no title worth showing: the stream's (or a file's) name is not one.
    assert all(p.title == "" for p in fallback_picks(six_hour_job, 6))


def test_a_fallback_pick_has_one_payoff_beat_at_the_peak(six_hour_job):
    from streamclip.judge.checks import _windows
    peaks = {n: w.hotspot.peak for n, w in _windows(six_hour_job).items()}
    for p in fallback_picks(six_hour_job, 6):
        assert [(b.kind, b.t) for b in p.beats] == [("payoff", peaks[p.candidate])]
        assert p.start <= p.beats[0].t <= p.end


def test_fallback_gives_fewer_when_fewer_are_ready_or_wanted(six_hour_job, tmp_path):
    assert len(fallback_picks(six_hour_job, 2)) == 2
    two = make_job(tmp_path / "two", [hotspot(1, 540.0), hotspot(2, 2000.0)], {})
    assert len(fallback_picks(two, 6)) == 2                        # and with no words at all
    assert fallback_picks(make_job(tmp_path / "none", [], {}), 6) == []


def test_a_short_stream_asks_for_fewer(three_minute_job):                            # Review Focus 5
    assert len(fallback_picks(three_minute_job, 6)) == 1


def test_a_peak_in_the_first_minute_still_makes_a_full_length_pick(three_minute_job):   # Review Focus 5
    job = make_job(three_minute_job.root.parent / "early",
                   [hotspot(1, 0.0, score=0.9, length=35.0, peak=5.0)], {1: spoken(0.0, 35.0)}, duration=180.0)
    pick = fallback_picks(job, 6)[0]
    assert pick.start >= 0.0 and pick.end <= 35.0 and (pick.end - pick.start) / 1.3 >= 15.0


def test_a_missing_job_gives_no_picks(tmp_path):
    from streamclip.jobdir import JobDir
    assert fallback_picks(JobDir(tmp_path / "nothing"), 6) == []
    assert apply_checks({"candidates": [verdict(1)]}, JobDir(tmp_path / "nothing"), 6) == []
