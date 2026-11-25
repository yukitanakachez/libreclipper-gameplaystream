"""Lane C, task 1: loudness against the stream's own normal, surges, and muted stretches."""

import numpy as np

from streamclip.signals import loudness


def test_one_value_per_second(synthetic_stream):
    db = loudness.per_second_db(synthetic_stream, 120.0)
    assert len(db) == 120 and db[31] > db[40] + 30


def test_sound_shorter_than_the_stream_is_padded_with_silence(synthetic_stream):
    db = loudness.per_second_db(synthetic_stream, 130.4)
    assert len(db) == 131 and db[119] > -50 and (db[121:] == -120.0).all()


def test_a_long_file_is_read_in_chunks(synthetic_stream, monkeypatch):
    monkeypatch.setattr(loudness, "CHUNK", 50)
    whole = loudness.per_second_db(synthetic_stream, 120.0)
    monkeypatch.undo()
    assert np.allclose(whole, loudness.per_second_db(synthetic_stream, 120.0), atol=0.5)


def test_relative_loudness_peaks_on_the_bursts(synthetic_stream):
    rel = loudness.relative_loudness(loudness.per_second_db(synthetic_stream, 120.0))
    assert rel.min() >= 0 and rel.max() <= 1
    assert rel[31] > 0.9 and rel[96] > 0.9 and rel[40] < 0.1


def test_a_stream_that_is_always_loud_has_no_peaks():
    assert loudness.relative_loudness(np.full(3600, -6.0)).max() < 0.1


def test_loudness_is_measured_against_the_surrounding_ten_minutes():
    db = np.concatenate([np.full(1800, -40.0), np.full(1800, -15.0)])
    db[900] = db[2700] = -10.0
    rel = loudness.relative_loudness(db)
    assert rel[900] == 1.0 and 0.3 < rel[2700] < 0.5      # +30 dB in the quiet half, +5 in the loud


def test_a_mostly_muted_stretch_does_not_make_room_noise_loud():
    db = np.full(1200, -120.0)
    db[500:520] = -62.0                                    # faint noise between two muted halves
    assert loudness.relative_loudness(db).max() == 0.0


def test_surge_fires_at_the_onset_only(synthetic_stream):
    s = loudness.surge(loudness.per_second_db(synthetic_stream, 120.0))
    assert s[30] > 0.8 and s[40] < 0.1
    assert s[33:50].max() == 0.0                           # the burst ending is not a surge


def test_sound_coming_back_after_silence_is_not_a_surge(synthetic_stream):
    s = loudness.surge(loudness.per_second_db(synthetic_stream, 120.0))
    assert s[78:84].max() < 0.1                            # 70-80 s is digital silence


def test_surge_on_nothing_and_on_one_second():
    assert len(loudness.surge(np.zeros(0))) == 0
    assert loudness.surge(np.array([-3.0]))[0] == 0.0
    assert len(loudness.relative_loudness(np.zeros(0))) == 0


def test_digital_silence_is_muted_after_thirty_seconds():
    db = np.concatenate([np.full(100, -30.0), np.full(60, -120.0), np.full(100, -30.0)])
    m = loudness.muted(db)
    assert m[130] and not m[50] and not m[105]        # the first 30 s of a silence are not yet "muted"
    assert not m[128] and m[129] and m[159] and not m[160]


def test_a_short_silence_and_a_silent_ending():
    db = np.concatenate([np.full(50, -30.0), np.full(20, -120.0), np.full(50, -30.0), np.full(40, -120.0)])
    m = loudness.muted(db)
    assert not m[:149].any() and m[149:].all()
