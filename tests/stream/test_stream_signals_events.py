"""Lane C, task 2: shouting, laughter and the like, from the sound classifier."""

import hashlib
import http.client
import io
import urllib.error

import numpy as np
import pytest

from streamclip.errors import StageError
from streamclip.signals import events


def fake_classify(scores_by_window):
    return lambda pcm: scores_by_window[: max(1, int(len(pcm) / 16000 / 0.975))]


def test_exciting_labels_become_the_voice_score(synthetic_stream):
    windows = [{"Speech": 0.9}] * 123
    windows[31] = {"Screaming": 0.8, "Speech": 0.3}
    windows[98] = {"Laughter": 0.6}
    track = events.voice_events(synthetic_stream, 120.0, classify=fake_classify(windows))
    assert len(track.voice) == 120
    assert track.voice[30] > 0.7 and track.label[30] == "Screaming"
    assert track.voice[95] > 0.5 and track.voice[10] == 0.0
    assert track.label[95] == "Laughter" and track.label[10] == ""
    assert len(track.music) == len(track.speech) == len(track.label) == 120


def test_music_and_speech_are_tracked_apart(synthetic_stream):
    track = events.voice_events(synthetic_stream, 120.0,
                                classify=fake_classify([{"Music": 0.9}] * 123))
    assert track.music.min() > 0.8 and track.speech.max() == 0.0


def test_the_strongest_exciting_label_names_the_second(synthetic_stream):
    windows = [{}] * 123
    windows[50] = {"Laughter": 0.4, "Shout": 0.7, "Speech": 0.99}       # 48.75 s to 49.725 s
    track = events.voice_events(synthetic_stream, 120.0, classify=fake_classify(windows))
    assert track.label[48] == track.label[49] == "Shout" and track.voice[49] == pytest.approx(0.7)
    assert track.label[47] == track.label[50] == ""


def test_a_long_stream_is_classified_in_chunks_that_line_up(synthetic_stream, monkeypatch):
    monkeypatch.setattr(events, "CHUNK_WINDOWS", 40)        # 39 s of sound at a time
    calls = []

    def classify(pcm):
        calls.append(len(pcm))
        loud = float(np.sqrt(np.mean(np.square(pcm[:15600])))) > 0.3
        return [{"Screaming": 1.0 if loud else 0.0}] + [{}] * (len(pcm) // 15600 - 1)

    track = events.voice_events(synthetic_stream, 120.0, classify=classify)
    assert len(calls) == 4 and calls[0] == 40 * 15600
    # Only a chunk starting inside a burst opens on a loud window. None does: chunks start
    # at 0, 39, 78 and 117 s, and the bursts are at 30-32 s and 95-97 s.
    assert track.voice.max() == 0.0 and len(track.voice) == 120


def test_scores_past_the_end_of_the_stream_are_dropped(synthetic_stream):
    track = events.voice_events(synthetic_stream, 20.0, classify=lambda pcm: [{"Gasp": 0.9}] * 500)
    assert len(track.voice) == 20 and track.voice.min() == pytest.approx(0.9)


def test_a_stream_with_no_sound_left_scores_nothing(synthetic_stream):
    track = events.voice_events(synthetic_stream, 300.0, classify=fake_classify([{"Cheering": 1.0}] * 400))
    assert len(track.voice) == 300 and track.voice[100] == 1.0 and track.voice[125:].max() == 0.0


MODEL = b"model bytes"


def refused(url):
    return urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)


@pytest.fixture
def download(tmp_path, monkeypatch):
    """Stands in for the network. `download.answers` holds what each request gets, in order:
    bytes to send back, or an error to raise. The last one is given to every later request.
    `download.fetched` and `download.waits` record what the engine asked for and how long it
    waited."""
    class Download:
        answers = [MODEL]
        fetched: list = []
        waits: list = []

    def fake_urlopen(url, timeout=None):
        answer = Download.answers[min(len(Download.fetched), len(Download.answers) - 1)]
        Download.fetched.append(url)
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(answer)

    Download.fetched, Download.waits = [], []
    monkeypatch.setenv("STREAMCLIP_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(events, "MODEL_SHA256", hashlib.sha256(MODEL).hexdigest())
    monkeypatch.setattr(events.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(events.time, "sleep", Download.waits.append)
    return Download


def test_the_model_is_downloaded_once(tmp_path, download):
    path = events.ensure_model()
    assert path == tmp_path / "models" / "yamnet.tflite" and path.read_bytes() == MODEL
    assert events.ensure_model() == path and download.fetched == [events.MODEL_URLS[0]]
    assert [p.name for p in path.parent.iterdir()] == ["yamnet.tflite"]
    assert download.waits == []


def test_the_real_model_is_the_same_file_at_both_addresses():
    # Both addresses are checked against one SHA-256, so either may answer.
    assert len(events.MODEL_URLS) == 2 and len(set(events.MODEL_URLS)) == 2
    assert all(url.startswith("https://") and url.endswith("/yamnet.tflite") for url in events.MODEL_URLS)
    assert events.MODEL_SHA256 == "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"


def test_a_refused_download_is_fetched_from_the_other_address(download):
    # A build machine was refused with "429 Too Many Requests", and the build stopped.
    download.answers = [refused(events.MODEL_URLS[0]), MODEL]
    assert events.ensure_model().read_bytes() == MODEL
    assert download.fetched == list(events.MODEL_URLS) and download.waits == []


def test_a_download_refused_everywhere_is_tried_again_after_a_wait(download):
    download.answers = [refused("first"), OSError("no network"), TimeoutError("slow"), MODEL]
    assert events.ensure_model().read_bytes() == MODEL
    assert download.fetched == list(events.MODEL_URLS) * 2
    assert download.waits == [events.WAITS[0]]


def test_a_download_cut_short_is_tried_again(download):
    download.answers = [http.client.IncompleteRead(b"mod"), MODEL]
    assert events.ensure_model().read_bytes() == MODEL


def test_a_file_that_is_not_the_model_is_not_kept(tmp_path, download):
    download.answers = [b"<html>Too Many Requests</html>", b"", MODEL]
    path = events.ensure_model()
    assert path.read_bytes() == MODEL and len(download.fetched) == 3
    assert [p.name for p in path.parent.iterdir()] == ["yamnet.tflite"]


def test_a_failed_download_leaves_nothing_behind(tmp_path, download):
    download.answers = [refused("anywhere")]
    with pytest.raises(StageError, match="sound classifier.*429.*Too Many Requests"):
        events.ensure_model()
    assert download.fetched == list(events.MODEL_URLS) * (len(events.WAITS) + 1)
    assert download.waits == list(events.WAITS)             # it does not wait after the last try
    assert list((tmp_path / "models").iterdir()) == []


def test_the_real_classifier_is_built_only_when_none_is_given(synthetic_stream, monkeypatch):
    built = []

    def default_classifier():
        built.append(1)
        return fake_classify([{"Applause": 0.5}] * 123)

    monkeypatch.setattr(events, "default_classifier", default_classifier)
    assert events.voice_events(synthetic_stream, 120.0).voice[60] == 0.5 and built == [1]
    events.voice_events(synthetic_stream, 120.0, classify=fake_classify([{}] * 123))
    assert built == [1]


@pytest.mark.live
def test_the_real_model_loads_and_runs(synthetic_stream):
    track = events.voice_events(synthetic_stream, 120.0)
    assert len(track.voice) == 120
