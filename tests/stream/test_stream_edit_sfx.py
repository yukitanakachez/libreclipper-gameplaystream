"""Where the sounds and the zoom go: on the judge's beats, sparingly."""

import numpy as np
import pytest

from clipper.pacing import TimeMap
from streamclip.edit import sfx
from streamclip.schema import Beat, Sound

PACK = {"payoff": Sound("boom", "/s/boom.wav", "payoff", -8.0, 0.0),
        "hard_cut": Sound("whoosh", "/s/whoosh.wav", "hard_cut", -12.0, 0.1),
        "reaction": Sound("pop", "/s/pop.wav", "reaction", -12.0, 0.0)}
FLAT = TimeMap([(0.0, 40.0)])


def speech_at(bursts, seconds: float = 40.0) -> np.ndarray:
    """Silence with noise at each (start, end, dB) in clip seconds."""
    rng = np.random.default_rng(3)
    pcm = np.zeros(int(seconds * 16000), np.float32)
    for start, end, db in bursts:
        a, b = int(start * 16000), int(end * 16000)
        pcm[a:b] = rng.uniform(-1.0, 1.0, b - a) * 10 ** (db / 20) * np.sqrt(3)
    return pcm


def test_a_beat_becomes_a_sound_in_cut_time():
    (p,) = sfx.place_sounds([Beat("payoff", 612.0)], 600.0, TimeMap([(0.0, 5.0), (7.0, 40.0)]), 1.3, PACK)
    assert (p.sound, p.path, p.at, p.gain_db) == ("boom", "/s/boom.wav", 10.0, -8.0)


def test_a_build_up_never_gets_a_sound_even_when_a_pack_offers_one():
    with_riser = PACK | {"build_up": Sound("riser", "/s/riser.wav", "build_up", -14.0, 1.2)}
    assert sfx.place_sounds([Beat("build_up", 610.0)], 600.0, FLAT, 1.3, PACK) == []
    assert sfx.place_sounds([Beat("build_up", 610.0)], 600.0, FLAT, 1.3, with_riser) == []


def test_a_sound_with_a_lead_starts_that_much_early():
    (p,) = sfx.place_sounds([Beat("hard_cut", 610.0)], 600.0, FLAT, 1.3, PACK)
    assert p.at == pytest.approx(9.9)


def test_a_sound_cannot_start_before_the_clip():
    assert sfx.place_sounds([Beat("hard_cut", 600.05)], 600.0, FLAT, 1.3, PACK)[0].at == 0.0


def test_never_more_than_five():
    beats = [Beat("reaction", 600.0 + 6 * i) for i in range(7)]
    assert len(sfx.place_sounds(beats, 600.0, TimeMap([(0.0, 60.0)]), 1.3, PACK)) == 5


def test_four_finished_seconds_apart_and_the_payoff_wins():
    beats = [Beat("reaction", 610.0), Beat("payoff", 612.0), Beat("reaction", 620.0)]
    placed = sfx.place_sounds(beats, 600.0, FLAT, 1.3, PACK)
    assert [p.sound for p in placed] == ["boom", "pop"] and placed[1].at - placed[0].at >= 4.0 * 1.3


def test_the_result_is_in_time_order():
    beats = [Beat("payoff", 630.0), Beat("reaction", 612.0), Beat("hard_cut", 620.0)]
    placed = sfx.place_sounds(beats, 600.0, FLAT, 1.3, PACK)
    assert [p.sound for p in placed] == ["pop", "whoosh", "boom"]


def test_unknown_beats_and_missing_sounds_are_skipped():
    assert sfx.place_sounds([Beat("dance", 610.0)], 600.0, FLAT, 1.3, PACK) == []
    assert sfx.place_sounds([Beat("payoff", 610.0)], 600.0, FLAT, 1.3, {}) == []


def test_beats_outside_the_clip_are_skipped():
    beats = [Beat("payoff", 590.0), Beat("reaction", 641.0)]
    assert sfx.place_sounds(beats, 600.0, FLAT, 1.3, PACK) == []
    assert sfx.place_sounds([Beat("payoff", 610.0)], 600.0, TimeMap([]), 1.3, PACK) == []


def test_zoom_goes_to_the_loudest_reaction():
    pcm = speech_at([(10.0, 10.5, -30.0), (20.0, 20.5, -6.0)])
    assert sfx.zoom_at([Beat("reaction", 610.2), Beat("reaction", 620.2), Beat("payoff", 630.0)], 600.0, pcm, 16000) == pytest.approx(20.2)


def test_zoom_falls_back_to_the_payoff_then_to_nothing():
    assert sfx.zoom_at([Beat("payoff", 630.0)], 600.0, np.zeros(16000 * 40, np.float32), 16000) == 30.0
    assert sfx.zoom_at([], 600.0, np.zeros(16000, np.float32), 16000) is None


def test_zoom_ignores_beats_outside_the_clip():
    pcm = speech_at([(10.0, 10.5, -6.0)])
    assert sfx.zoom_at([Beat("reaction", 590.0), Beat("reaction", 700.0)], 600.0, pcm, 16000) is None
    assert sfx.zoom_at([Beat("reaction", 700.0), Beat("payoff", 610.0)], 600.0, pcm, 16000) == 10.0
