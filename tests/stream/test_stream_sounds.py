"""The sound pack: four synthesised sounds, one per beat kind, and where each came from."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from streamclip.sounds import load_pack, make_pack

ASSETS = Path(make_pack.ASSETS)
MANIFEST = ASSETS / "manifest.json"


def _samples(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    return x, rate


def _part(name, first_third=False, last_third=False) -> tuple[np.ndarray, int]:
    x, rate = _samples(ASSETS / name)
    third = len(x) // 3
    if first_third:
        x = x[:third]
    if last_third:
        x = x[-third:]
    return x, rate


def seconds(path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def peak_dbfs(path) -> float:
    x, _ = _samples(path)
    return 20 * np.log10(max(np.abs(x).max(), 1e-12))


def centroid_hz(name, first_third=False, last_third=False) -> float:
    x, rate = _part(name, first_third, last_third)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    return float((freqs * power).sum() / power.sum())


def rms_db(name, first_third=False, last_third=False) -> float:
    x, _ = _part(name, first_third, last_third)
    return 20 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1e-12))


def test_every_beat_kind_has_a_sound():
    pack = load_pack()
    assert set(pack) == {"payoff", "hard_cut", "reaction"}
    assert {s.id for s in pack.values()} == {"boom", "whoosh", "pop"}
    assert all(Path(s.path).is_absolute() and Path(s.path).exists() for s in pack.values())
    assert all(role == s.role for role, s in pack.items())


def test_files_are_48k_mono_16_bit_and_short():
    limits = {"boom": (0.6, 1.2), "whoosh": (0.25, 0.6), "pop": (0.05, 0.3)}
    for s in load_pack().values():
        with wave.open(s.path) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (48000, 1, 2)
            assert limits[s.id][0] <= w.getnframes() / 48000 <= limits[s.id][1]


def test_nothing_clips():
    for s in load_pack().values():
        assert peak_dbfs(s.path) <= -1.0


def test_there_is_no_riser():
    # Yuki's call: no rising sound before a moment, anywhere.
    assert "build_up" not in load_pack()
    assert not (ASSETS / "riser.wav").exists()
    assert "riser" not in MANIFEST.read_text()
    assert not hasattr(make_pack, "riser")


def test_sounds_land_on_the_beat_and_the_whoosh_just_before_it():
    pack = load_pack()
    assert pack["payoff"].lead == 0.0 and pack["reaction"].lead == 0.0 and pack["hard_cut"].lead == 0.1


def test_gains_keep_sounds_under_the_voice():
    assert all(-16.0 <= s.gain_db <= -6.0 for s in load_pack().values())


def test_each_sound_has_its_character():
    assert centroid_hz("boom.wav") < 250                              # a low thud
    assert centroid_hz("whoosh.wav", last_third=True) > 2 * centroid_hz("whoosh.wav", first_third=True)


def test_the_manifest_says_where_each_sound_came_from():
    rows = json.loads(MANIFEST.read_text())
    assert len(rows) == 3
    for row in rows:
        assert row["source"] == "synthesised by streamclip/sounds/make_pack.py" and row["licence"] == "CC0-1.0"
        assert set(row) >= {"id", "file", "role", "gain_db", "lead", "seconds", "source", "licence"}


def test_the_generator_reproduces_the_files_within_two_steps(tmp_path):
    # Another machine's sine and exponential can differ in the last bit and move a sample by
    # a step of the 16-bit scale, so the sound is compared sample by sample, not byte by byte.
    make_pack.main(tmp_path)
    for name in ("boom.wav", "whoosh.wav", "pop.wav"):
        with wave.open(str(tmp_path / name)) as new, wave.open(str(ASSETS / name)) as old:
            assert new.getframerate() == old.getframerate()
            assert new.getnframes() == old.getnframes()
            a = np.frombuffer(new.readframes(new.getnframes()), dtype="<i2").astype(np.int32)
            b = np.frombuffer(old.readframes(old.getnframes()), dtype="<i2").astype(np.int32)
        assert np.abs(a - b).max() <= 2
    assert (tmp_path / "manifest.json").read_bytes() == MANIFEST.read_bytes()
