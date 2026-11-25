"""Lane C, task 3: chat bursts and the clips viewers already made."""

import numpy as np

from streamclip.schema import ViewerClip
from streamclip.signals import extras


def test_a_burst_shows_up_a_few_seconds_early():
    times = [float(t) for t in range(0, 1200)]                 # one message a second
    times += [600.0 + i / 20 for i in range(100)]              # a burst from 600 s to 605 s
    burst = extras.chat_burst(times, 1200.0)
    assert len(burst) == 1200 and burst.min() >= 0 and burst.max() <= 1
    assert int(np.argmax(burst)) in range(592, 600) and burst[300] < 0.1


def test_steady_busy_chat_is_not_a_burst():
    assert extras.chat_burst([t / 10 for t in range(12000)], 1200.0).max() < 0.1


def test_no_chat_gives_none():
    assert extras.chat_burst(None, 600.0) is None and extras.chat_burst([], 600.0) is None
    assert extras.chat_ratio(None, 600.0) is None


def test_the_ratio_says_how_many_times_the_normal_rate():
    times = [float(t) for t in range(0, 1200)] + [900.0 + i / 3 for i in range(60)]   # 4 a second for 20 s
    ratio = extras.chat_ratio(times, 1200.0)
    assert 3.9 < ratio[900] < 4.1 and 0.9 < ratio[300] < 1.1   # 900 here is 906 s of chat


def test_a_few_messages_in_a_slow_chat_are_not_a_burst():
    times = [float(t) for t in range(0, 3600, 60)] + [1800.2, 1800.9, 1801.5]
    assert extras.chat_burst(times, 3600.0).max() == 0.0


def test_chat_at_the_very_start_is_measured_against_the_first_minute():
    burst = extras.chat_burst([t / 5 for t in range(3000)], 600.0)          # 5 a second from 0 s
    assert burst.max() == 0.0


def test_messages_outside_the_stream_are_ignored():
    burst = extras.chat_burst([-5.0, 700.0, 10.0, 10.5, float("nan")], 600.0)
    assert len(burst) == 600 and burst.max() == 0.0


def test_clips_bump_their_own_span_by_views():
    bump = extras.clip_bump([ViewerClip(100.0, 30.0, 12000, "a"), ViewerClip(400.0, 20.0, 10, "b")], 600.0)
    assert bump[110] == 1.0 and 0 < bump[410] < 0.5 and bump[300] == 0.0
    assert bump[99] == 0.0 and bump[100] == 1.0 and bump[129] == 1.0 and bump[130] == 0.0


def test_a_clip_past_the_end_is_ignored():
    assert extras.clip_bump([ViewerClip(900.0, 30.0, 5, "a")], 600.0).max() == 0.0


def test_no_clips_gives_none():
    assert extras.clip_bump(None, 600.0) is None and extras.clip_bump([], 600.0) is None
    assert extras.clip_views(None, 600.0) is None


def test_overlapping_clips_add_up_to_one_at_most():
    clips = [ViewerClip(100.0, 30.0, 1000, "a"), ViewerClip(110.0, 30.0, 1000, "b"),
             ViewerClip(300.0, 10.0, 30, "c"), ViewerClip(305.0, 10.0, 30, "d")]
    bump = extras.clip_bump(clips, 600.0)
    assert bump.max() == 1.0 and bump[115] == 1.0
    assert bump[307] == 2 * bump[302] and 0.9 < bump[307] < 1.0


def test_clips_nobody_watched_still_count_alike():
    bump = extras.clip_bump([ViewerClip(100.0, 30.0, 0, "a"), ViewerClip(595.0, 30.0, 0, "b")], 600.0)
    assert bump[110] == 1.0 and bump[599] == 1.0 and len(bump) == 600


def test_views_per_second_take_the_most_watched_clip():
    views = extras.clip_views([ViewerClip(100.0, 30.0, 12000, "a"), ViewerClip(120.0, 30.0, 40, "b")], 600.0)
    assert views[125] == 12000 and views[140] == 40 and views[10] == 0


def test_a_stream_of_a_few_seconds():
    assert len(extras.chat_burst([0.5, 1.5], 3.0)) == 3
    assert len(extras.clip_bump([ViewerClip(1.0, 30.0, 5, "a")], 3.0)) == 3
