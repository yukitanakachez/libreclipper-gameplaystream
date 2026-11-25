import numpy as np

from clipper import shell
from streamclip import audio


def test_read_pcm_slices(synthetic_stream):
    pcm = audio.read_pcm(synthetic_stream, start=30.0, duration=2.0)
    assert pcm.dtype == np.float32 and abs(len(pcm) - 32000) < 400


def test_fixture_levels(synthetic_stream):
    loud = audio.level_db(audio.read_pcm(synthetic_stream, 30.0, 2.0))
    tone = audio.level_db(audio.read_pcm(synthetic_stream, 10.0, 4.0))
    quiet = audio.level_db(audio.read_pcm(synthetic_stream, 40.0, 4.0))
    dead = audio.level_db(audio.read_pcm(synthetic_stream, 72.0, 4.0))
    assert loud > tone + 10 and tone > quiet + 10 and dead < -80


def test_fixture_matches_the_plan(synthetic_stream):
    video, sound = (next(s for s in shell.probe(synthetic_stream)["streams"] if s["codec_type"] == kind)
                    for kind in ("video", "audio"))
    assert (video["width"], video["height"], video["r_frame_rate"]) == (1280, 720, "30/1")
    assert (sound["codec_name"], sound["channels"], sound["sample_rate"]) == ("aac", 2, "48000")
    assert abs(float(shell.probe(synthetic_stream)["format"]["duration"]) - 120.0) < 0.1
    for start, length, level in ((40.0, 4.0, -40.0), (10.5, 3.0, -20.0), (21.0, 4.0, -20.0),
                                 (51.0, 6.0, -20.0), (30.25, 1.5, -3.0), (95.25, 1.5, -3.0),
                                 (60.0, 5.0, -40.0), (100.0, 10.0, -40.0)):
        assert abs(audio.level_db(audio.read_pcm(synthetic_stream, start, length)) - level) < 1.0


def test_read_pcm_to_the_end_and_past_it(synthetic_stream):
    assert abs(len(audio.read_pcm(synthetic_stream, start=115.0)) - 5 * 16000) < 800
    assert len(audio.read_pcm(synthetic_stream, start=130.0, duration=2.0)) == 0
    assert abs(len(audio.read_pcm(synthetic_stream, 0.0, 1.0, rate=8000)) - 8000) < 200


def test_levels_db_one_value_per_hop(synthetic_stream):
    values = audio.levels_db(audio.read_pcm(synthetic_stream, 0.0, 10.0), 16000, 0.05)
    assert abs(len(values) - 200) <= 1


def test_levels_db_follows_the_sound():
    rate = 16000
    pcm = np.concatenate([np.zeros(rate, np.float32), np.full(rate, 0.1, np.float32)])
    values = audio.levels_db(pcm, rate, 0.25)
    assert len(values) == 8
    assert np.allclose(values[:4], -120.0) and np.allclose(values[4:], -20.0, atol=0.01)


def test_levels_db_keeps_time_when_a_hop_is_not_whole_samples():
    values = audio.levels_db(np.ones(16000 * 60, np.float32), 16000, 1 / 30)
    assert len(values) == 1800 and np.allclose(values, 0.0, atol=0.01)


def test_levels_db_of_nothing():
    assert len(audio.levels_db(np.zeros(0, np.float32), 16000, 0.05)) == 0


def test_silence_is_minus_120():
    assert audio.level_db(np.zeros(1600, dtype=np.float32)) == -120.0
    assert audio.level_db(np.zeros(0, dtype=np.float32)) == -120.0


def test_fixture_picture_moves_then_freezes(synthetic_stream, mean_frame_difference):
    assert mean_frame_difference(synthetic_stream, 20.0, 26.0) > 4.0
    assert mean_frame_difference(synthetic_stream, 72.0, 78.0) < 0.5
