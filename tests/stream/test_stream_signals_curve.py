"""Lane C, task 4: the excitement curve and the shortlist of hot spots."""

import json

import numpy as np
import pytest

from streamclip.schema import Hotspot, Signals, ViewerClip
from streamclip.signals import build_signals, curve, ensure_model, events, pick_hotspots


def signals(excitement, details=None, **parts):
    """A Signals with flat parts, one value per second of `excitement`."""
    count = len(excitement)
    flat = {name: [0.0] * count for name in ("loudness", "surge", "voice", "penalty")}
    flat.update(chat=None, clips=None)
    flat.update({name: [float(v) for v in values] for name, values in parts.items()})
    sig = Signals(duration=float(count), excitement=[float(v) for v in excitement], **flat)
    if details is not None:
        sig.details = details
    return sig


def spike_at(second, count, height=0.9):
    exc = np.zeros(count)
    exc[second] = height
    return exc


def quiet_classifier(pcm):
    return [{"Speech": 0.0}] * (len(pcm) // 15600 + 1)


def music_classifier(pcm):                                 # music with nobody talking, 20 s to 45 s
    return [{"Music": 0.9, "Speech": 0.05} if 20 <= i * 0.975 < 45 else {}
            for i in range(len(pcm) // 15600 + 1)]


# --- the curve ---------------------------------------------------------------------------

def test_weights_are_shared_out_again_when_extras_are_missing(synthetic_stream):
    sig = build_signals(synthetic_stream, 120.0, None, None, classify=quiet_classifier)
    assert sig.chat is None and sig.clips is None
    # With nothing from the classifier the most a second can reach is loudness plus surge:
    # their share of the weight as written, once chat's and clips' share is handed round
    # (0.25 of 0.64, about 0.39, with the weights tuned on real streams).
    weights = curve.WEIGHTS
    sound = weights["loudness"] + weights["surge"]
    top = max(sig.excitement)
    assert top == pytest.approx(sound / (sound + weights["voice"]), abs=1e-3)
    assert sig.excitement[30] == top and sig.excitement[95] == top
    assert sig.excitement[40] < 0.01 and sig.excitement[75] == 0.0 and sig.excitement[80] < 0.01


def test_every_curve_has_one_value_per_second_and_goes_to_json(synthetic_stream):
    chat = [float(t) for t in range(120)] + [60.0 + i / 20 for i in range(100)]
    sig = build_signals(synthetic_stream, 120.0, chat, [ViewerClip(90.0, 20.0, 500, "a")],
                        classify=quiet_classifier)
    for name in ("loudness", "surge", "voice", "chat", "clips", "penalty", "excitement"):
        values = getattr(sig, name)
        assert len(values) == 120 and all(type(v) is float and 0.0 <= v <= 1.0 for v in values), name
    assert sig.duration == 120.0 and sig.clips[95] == 1.0 and max(sig.chat) > 0.5
    back = Signals.from_dict(json.loads(json.dumps(sig.to_dict())))
    assert back == sig
    # Loudness and surge carry their own weight now, and the clip at 95 s adds its own.
    sound = curve.WEIGHTS["loudness"] + curve.WEIGHTS["surge"]
    assert sig.excitement[30] == pytest.approx(sound, abs=1e-3)
    assert sig.excitement[95] == pytest.approx(sound + curve.WEIGHTS["clips"], abs=1e-3)


def test_the_weights_add_up_to_one():
    assert sum(curve.WEIGHTS.values()) == pytest.approx(1.0)
    assert set(curve.WEIGHTS) == {"loudness", "surge", "voice", "chat", "clips"}


def test_combine_shares_a_missing_signals_weight_in_proportion():
    one, none = np.ones(3), np.zeros(3)
    parts = {"loudness": one, "surge": none, "voice": none, "chat": None, "clips": None}
    w = curve.WEIGHTS
    sound = w["loudness"] + w["surge"] + w["voice"]
    assert curve.combine(parts, none) == pytest.approx([w["loudness"] / sound] * 3)
    parts.update(chat=none)
    assert curve.combine(parts, none) == pytest.approx([w["loudness"] / (sound + w["chat"])] * 3)
    parts.update(surge=one, voice=one, chat=one, clips=one)
    assert curve.combine(parts, np.array([0.0, 0.5, 1.0])) == pytest.approx([1.0, 0.5, 0.0])


def test_music_without_speech_is_marked_down(synthetic_stream):
    plain = build_signals(synthetic_stream, 120.0, None, None, classify=quiet_classifier)
    music = build_signals(synthetic_stream, 120.0, None, None, classify=music_classifier)
    assert music.penalty[30] == pytest.approx(0.9) and plain.penalty[30] == 0.0
    assert 0 < music.excitement[30] < plain.excitement[30] / 2
    assert music.excitement[95] == plain.excitement[95] and music.penalty[60] == 0.0


def test_music_counts_only_when_it_lasts_and_nobody_talks():
    music, speech = np.zeros(200), np.zeros(200)
    music[10:29] = 0.9                                      # 19 s: too short
    music[50:70] = 0.8                                      # 20 s: an intermission
    music[100:140] = 0.9
    speech[100:140] = 0.5                                   # talking over music: a normal stream
    music[150:190] = 0.55                                   # faint music
    pen = curve.penalty(np.zeros(200, dtype=bool), music, speech)
    assert pen[10:29].max() == 0.0 and (pen[50:70] == 0.8).all() and pen[70:].max() == 0.0


def test_muted_sound_is_marked_all_the_way_down():
    muted = np.zeros(100, dtype=bool)
    muted[40:60] = True
    pen = curve.penalty(muted, np.zeros(100), np.zeros(100))
    assert (pen[40:60] == 1.0).all() and pen.sum() == 20.0


def test_the_real_classifier_is_looked_up_when_none_is_given(synthetic_stream, monkeypatch):
    monkeypatch.setattr(events, "default_classifier", lambda: lambda pcm: [{"Screaming": 1.0}] * 200)
    sig = build_signals(synthetic_stream, 120.0, None, None)
    assert sig.voice[60] == 1.0 and sig.excitement[30] == 1.0


def test_the_package_hands_on_the_three_names():
    assert ensure_model is events.ensure_model


# --- the shortlist -----------------------------------------------------------------------

def test_peaks_are_45_seconds_apart_and_capped():
    exc = np.zeros(7200); exc[[100, 120, 130, 1000, 2000, 3000]] = [.9, .8, .95, .7, .6, .5]
    hot = pick_hotspots(signals(exc), wanted=1)
    assert len(hot) == 4 and [h.n for h in hot] == [1, 2, 3, 4]
    peaks = sorted(h.peak for h in hot)
    assert all(b - a >= 45 for a, b in zip(peaks, peaks[1:])) and 130.0 in peaks and 120.0 not in peaks
    assert [h.peak for h in hot] == peaks                   # numbered in time order


def test_the_strongest_peaks_win_when_there_are_too_many():
    exc = np.zeros(3600); exc[[300, 900, 1500, 2100, 2700, 3300]] = [.2, .9, .3, .8, .7, .6]
    hot = pick_hotspots(signals(exc), wanted=1)
    assert [h.peak for h in hot] == [900.0, 2100.0, 2700.0, 3300.0]
    assert hot[0].score > hot[1].score > hot[2].score > hot[3].score > 0


def test_window_is_sixty_before_and_thirty_after():
    h = pick_hotspots(signals(spike_at(600, 7200)), 1)[0]
    assert (h.start, h.peak, h.end) == (540.0, 600.0, 630.0)


def test_windows_clamp_to_the_stream():                   # Review Focus 5
    hot = pick_hotspots(signals(spike_at(20, 180)), 6)
    assert hot and hot[0].start == 0.0 and all(h.end <= 180.0 for h in hot) and len(hot) <= 4
    last = pick_hotspots(signals(spike_at(175, 180)), 6)[0]
    assert (last.start, last.peak, last.end) == (115.0, 175.0, 180.0)


def test_a_stream_shorter_than_one_window():               # Review Focus 5
    hot = pick_hotspots(signals(spike_at(10, 40)), 6)
    assert [(h.n, h.start, h.peak, h.end) for h in hot] == [(1, 0.0, 10.0, 40.0)]
    assert pick_hotspots(signals(np.zeros(0)), 6) == []


def test_no_half_hour_takes_more_than_a_third():
    exc = np.zeros(4 * 3600); exc[100:1700:50] = 0.9; exc[5000::400] = 0.5
    hot = pick_hotspots(signals(exc), 6)
    assert sum(1 for h in hot if h.peak < 1800) == 8        # 24 wanted, a third is 8
    assert len(hot) == 24


def test_a_stream_under_ninety_minutes_is_not_spread():
    exc = np.zeros(5000); exc[100:1700:50] = 0.9; exc[3000::400] = 0.5
    hot = pick_hotspots(signals(exc), 6)
    assert len(hot) == 24 and sum(1 for h in hot if h.peak < 1800) == 24


def test_each_hot_spot_says_what_fired():
    count = 7200
    voice, chat, loud = np.zeros(count), np.zeros(count), np.zeros(count)
    voice[601], chat[602], loud[598] = 0.8, 2 / 3, 0.4        # loudness is under a half: not listed
    details = curve.Details(label=[""] * count, over_db=np.zeros(count), rise_db=np.zeros(count),
                            chat_ratio=np.zeros(count), clip_views=None)
    details.label[601], details.chat_ratio[602] = "Screaming", 4.0
    hot = pick_hotspots(signals(spike_at(600, count), details, voice=voice, chat=chat, loudness=loud), 1)
    fired = hot[0].fired
    assert [(f.signal, f.t, f.detail) for f in fired] == [("voice", 601.0, "Screaming"), ("chat", 602.0, "chat ×4")]
    assert fired[0].strength == pytest.approx(0.8) and fired[1].strength == pytest.approx(0.667, abs=1e-3)
    assert Hotspot.from_dict(json.loads(json.dumps(hot[0].to_dict()))) == hot[0]


def test_every_signal_has_its_own_wording():
    count = 3600
    parts = {name: np.zeros(count) for name in ("loudness", "surge", "voice", "chat", "clips")}
    parts["loudness"][300], parts["surge"][299], parts["clips"][303] = 1.0, 0.9, 0.7
    details = curve.Details(label=[""] * count, over_db=np.full(count, 14.2), rise_db=np.full(count, 16.8),
                            chat_ratio=None, clip_views=np.full(count, 12400))
    fired = pick_hotspots(signals(spike_at(300, count), details, **parts), 1)[0].fired
    assert [(f.signal, f.detail) for f in fired] == [
        ("loudness", "+14 dB over the ten minutes around it"),
        ("surge", "jumped 17 dB within 3 seconds"),
        ("viewer_clip", "viewer clip, 12k views")]


def test_signals_only_fire_within_five_seconds_of_the_peak():
    voice = np.zeros(3600); voice[306] = 0.9; voice[295] = 0.7
    details = curve.Details(label=["Laughter"] * 3600, over_db=np.zeros(3600), rise_db=np.zeros(3600),
                            chat_ratio=None, clip_views=None)
    fired = pick_hotspots(signals(spike_at(300, 3600), details, voice=voice), 1)[0].fired
    assert [(f.signal, f.t) for f in fired] == [("voice", 295.0)]


def test_signals_read_back_from_a_file_still_say_what_fired():
    count = 3600
    parts = {name: np.zeros(count) for name in ("loudness", "surge", "voice", "chat", "clips")}
    for name, value in (("loudness", 1.0), ("surge", 0.75), ("voice", 0.8), ("chat", 2 / 3), ("clips", 0.6)):
        parts[name][300] = value
    sig = Signals.from_dict(signals(spike_at(300, count), **parts).to_dict())
    fired = pick_hotspots(sig, 1)[0].fired
    assert {f.signal: f.detail for f in fired} == {
        "loudness": "+12 dB or more over the ten minutes around it",
        "voice": "shouting, laughter or cheering",
        "surge": "jumped 15 dB within 3 seconds",
        "chat": "chat ×4",
        "viewer_clip": "viewer clip"}
    assert [f.strength for f in fired] == sorted((f.strength for f in fired), reverse=True)


def test_a_flat_stream_still_returns_hot_spots():
    noise = np.random.default_rng(7).uniform(0, 0.01, 7200)
    hot = pick_hotspots(signals(np.full(7200, 0.05) + noise), 6)
    assert len(hot) == 24 and all(h.fired == [] for h in hot)
    assert all(b.peak - a.peak >= 45 for a, b in zip(hot, hot[1:]))


def test_a_silent_stream_has_no_hot_spots():
    assert pick_hotspots(signals(np.zeros(7200)), 6) == []
    assert pick_hotspots(signals(spike_at(300, 3600)), 0) == []


def test_the_test_stream_shortlists_both_loud_moments(synthetic_stream):
    sig = build_signals(synthetic_stream, 120.0, None, None, classify=quiet_classifier)
    hot = pick_hotspots(sig, 6)
    assert 1 <= len(hot) <= 3 and all(0.0 <= h.start <= h.peak <= h.end <= 120.0 for h in hot)
    for burst in (30.0, 95.0):
        assert any(h.start <= burst and burst + 2 <= h.end for h in hot)
    assert any(f.signal == "loudness" for h in hot for f in h.fired)
