"""Lane D, task 1: one window transcribed with word timings, in stream seconds."""

from types import SimpleNamespace

import pytest

from streamclip.windows import transcribe
from streamclip.windows.transcribe import captions_allowed, transcribe_window


def seg(start, end, words):
    """A faster-whisper style segment; word times are window seconds."""
    return SimpleNamespace(
        start=start, end=end, text=" " + " ".join(w for w, _, _ in words),
        words=[SimpleNamespace(word=" " + w, start=a, end=b, probability=0.9) for w, a, b in words])


class FakeModel:
    """Has the same `transcribe` shape as faster_whisper.WhisperModel."""

    def __init__(self, segments, language="en"):
        self.segments = segments
        self.language = language
        self.seen_language = "unset"
        self.seen_samples = None

    def transcribe(self, pcm, language=None, word_timestamps=False, vad_filter=False, **_):
        assert word_timestamps and vad_filter
        self.seen_language = language
        self.seen_samples = len(pcm)
        info = SimpleNamespace(language=language or self.language, language_probability=0.99)
        return iter(self.segments), info


def test_word_times_are_moved_to_stream_seconds(synthetic_stream):
    model = FakeModel([seg(1.0, 2.0, [("oh", 1.0, 1.3), ("no!", 1.4, 2.0)])], language="en")
    transcript, language = transcribe_window(synthetic_stream, 600.0, 690.0, model=model)
    assert language == "en"
    assert [(w.i, w.text, w.start, w.end) for w in transcript.words] == [
        (0, "oh", 601.0, 601.3), (1, "no!", 601.4, 602.0)]
    assert transcript.segments[0].first_word == 0 and transcript.segments[0].last_word == 1
    assert transcript.segments[0].start == 601.0 and transcript.segments[0].text == "oh no!"
    assert transcript.duration == 690.0


def test_the_window_sound_is_what_the_model_hears(synthetic_stream):
    model = FakeModel([], language="en")
    transcribe_window(synthetic_stream, 20.0, 30.0, model=model)
    assert abs(model.seen_samples - 10 * 16000) < 160


def test_a_window_with_no_speech_gives_an_empty_transcript(synthetic_stream):     # Review Focus 1
    transcript, language = transcribe_window(synthetic_stream, 70.0, 80.0,
                                             model=FakeModel([], language="en"))
    assert transcript.words == [] and transcript.segments == [] and transcript.duration == 80.0
    assert language == "en"


def test_words_with_no_text_or_no_times_are_left_out(synthetic_stream):
    s = seg(1.0, 2.0, [("a", 1.0, 1.2), ("b", 1.3, 1.5)])
    s.words[0].word = "  "
    s.words[1].start = None
    empty = seg(3.0, 4.0, [])
    good = seg(5.0, 6.0, [("yes", 5.0, 5.5)])
    transcript, _ = transcribe_window(synthetic_stream, 0.0, 10.0, model=FakeModel([s, empty, good]))
    assert [(w.i, w.text) for w in transcript.words] == [(0, "yes")]
    assert len(transcript.segments) == 1 and transcript.segments[0].first_word == 0


def raw_seg(start, end, tokens):
    """A segment from the model's own tokens, leading spaces and all: "-term" comes with none."""
    return SimpleNamespace(
        start=start, end=end, text="".join(t for t, _, _ in tokens),
        words=[SimpleNamespace(word=t, start=a, end=b, probability=0.9) for t, a, b in tokens])


def test_a_word_that_starts_with_a_hyphen_joins_the_word_before(synthetic_stream):
    model = FakeModel([raw_seg(1.0, 3.0, [(" long", 1.0, 1.3), ("-term", 1.3, 1.7),
                                          (" desire", 1.8, 2.4)])])
    transcript, _ = transcribe_window(synthetic_stream, 600.0, 690.0, model=model)
    assert [(w.i, w.text, w.start, w.end) for w in transcript.words] == [
        (0, "long-term", 601.0, 601.7), (1, "desire", 601.8, 602.4)]
    assert (transcript.segments[0].first_word, transcript.segments[0].last_word) == (0, 1)


def test_a_word_that_starts_with_an_apostrophe_joins_the_word_before(synthetic_stream):
    model = FakeModel([raw_seg(1.0, 3.0, [(" I", 1.0, 1.2), ("'m", 1.2, 1.4), (" done", 1.5, 2.0),
                                          (" rock", 2.1, 2.3), ("’n", 2.3, 2.4)])])
    transcript, _ = transcribe_window(synthetic_stream, 0.0, 10.0, model=model)
    assert [w.text for w in transcript.words] == ["I'm", "done", "rock’n"]
    assert transcript.words[0].end == 1.4 and [w.i for w in transcript.words] == [0, 1, 2]


def test_a_split_word_is_joined_across_segments(synthetic_stream):
    model = FakeModel([raw_seg(1.0, 2.0, [(" short", 1.0, 1.6)]),
                       raw_seg(1.6, 3.0, [("-lived", 1.6, 2.0), (" fun", 2.2, 2.5)])])
    transcript, _ = transcribe_window(synthetic_stream, 0.0, 10.0, model=model)
    assert [(w.text, w.end) for w in transcript.words] == [("short-lived", 2.0), ("fun", 2.5)]
    assert [(s.first_word, s.last_word) for s in transcript.segments] == [(0, 0), (1, 1)]


def test_a_first_word_with_a_hyphen_has_nothing_to_join(synthetic_stream):
    model = FakeModel([raw_seg(1.0, 2.0, [("'cause", 1.0, 1.4), (" yes", 1.5, 1.9)])])
    transcript, _ = transcribe_window(synthetic_stream, 0.0, 10.0, model=model)
    assert [w.text for w in transcript.words] == ["'cause", "yes"]


def test_the_second_pass_hears_whole_words_too(synthetic_stream):
    model = FakeModel([raw_seg(1.0, 3.0, [(" long", 1.0, 1.3), ("-term", 1.3, 1.7)])])
    words = transcribe.transcribe_clip(synthetic_stream, 5.0, 20.0, 120.0, model=model)
    assert [w.text for w in words] == ["long-term"]


def test_a_given_language_is_passed_through(synthetic_stream):
    model = FakeModel([], language="en")
    transcribe_window(synthetic_stream, 0.0, 10.0, language="es", model=model)
    assert model.seen_language == "es"


def test_no_model_means_the_default_model(synthetic_stream, monkeypatch):
    fake = FakeModel([], language="en")
    monkeypatch.setattr(transcribe, "default_model", lambda: fake)
    transcribe_window(synthetic_stream, 0.0, 10.0)
    assert fake.seen_samples is not None


def test_ensure_model_builds_the_default_model(monkeypatch):
    calls = []
    monkeypatch.setattr(transcribe, "default_model", lambda: calls.append("first"))
    monkeypatch.setattr(transcribe, "caption_model", lambda: calls.append("second"))
    transcribe.ensure_model()
    assert calls == ["first", "second"]


@pytest.mark.parametrize("text, allowed", [
    ("oh no he did not", True), ("¿qué está pasando aquí?", True),
    ("これは何ですか", False), ("что происходит", False), ("", True),
    ("LET'S GO 123 !!! 😂😂", True), ("ok ok ok ok ok ok ok これ", True)])
def test_captions_only_for_latin_letters(text, allowed):
    assert captions_allowed(text) is allowed


@pytest.mark.live
def test_the_real_model_hears_nothing_in_silence(synthetic_stream):
    transcript, _ = transcribe_window(synthetic_stream, 70.0, 80.0, language="en")
    assert transcript.words == []


def test_the_speech_model_uses_the_processors_this_job_may_use(monkeypatch):
    import sys
    from types import ModuleType

    from streamclip import settings
    built = []
    fake = ModuleType("faster_whisper")
    fake.WhisperModel = lambda name, **how: built.append((name, how)) or object()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(settings, "processors", lambda: 4)
    monkeypatch.setattr(transcribe, "_models", {})
    transcribe.default_model()
    assert built[0][1]["cpu_threads"] == 4 and built[0][1]["device"] == "cpu"
    # More than eight made it slower on a machine whose extra processors are the slow kind.
    monkeypatch.setattr(settings, "processors", lambda: 14)
    monkeypatch.setattr(transcribe, "_models", {})
    transcribe.default_model()
    assert built[1][1]["cpu_threads"] == 8


def test_each_pass_has_its_own_model_built_once(monkeypatch):
    import sys
    from types import ModuleType

    from streamclip import settings
    built = []
    fake = ModuleType("faster_whisper")
    fake.WhisperModel = lambda name, **how: built.append(name) or object()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(transcribe, "_models", {})
    assert transcribe.default_model() is transcribe.default_model()
    assert transcribe.caption_model() is not transcribe.default_model()
    assert built == [settings.WINDOW_MODEL, settings.CAPTION_MODEL] == ["base", "small"]


class Listening(FakeModel):
    def transcribe(self, pcm, beam_size=None, **how):
        self.seen_beam = beam_size
        return super().transcribe(pcm, **how)


def test_the_first_pass_is_fast_and_is_handed_samples_not_a_file(synthetic_stream):
    import numpy as np
    model = Listening([])
    seen = []
    real = model.transcribe
    model.transcribe = lambda pcm, **how: seen.append(pcm) or real(pcm, **how)
    transcribe_window(synthetic_stream, 0.0, 10.0, model=model)
    assert model.seen_beam == 1
    assert isinstance(seen[0], np.ndarray) and seen[0].dtype == np.float32      # PyAV 19 breaks file paths


def test_the_second_pass_hears_one_clip_accurately_with_a_second_either_side(synthetic_stream):
    # Word times are seconds into what the model heard, which starts a second before the clip.
    model = Listening([seg(0.2, 1.4, [("before", 0.2, 0.6), ("edge", 0.8, 1.4)]),
                       seg(5.0, 6.0, [("inside", 5.0, 6.0)]),
                       seg(11.2, 11.9, [("after", 11.2, 11.9)])])
    words = transcribe.transcribe_clip(synthetic_stream, 30.0, 40.0, 120.0, language="en", model=model)
    assert model.seen_beam == 5 and model.seen_language == "en"
    assert model.seen_samples == pytest.approx(12 * 16000, abs=1600)
    assert [(w.i, w.text, w.start) for w in words] == [(0, "edge", 29.8), (1, "inside", 34.0)]


def test_the_second_pass_stays_inside_the_stream(synthetic_stream):
    model = Listening([])
    transcribe.transcribe_clip(synthetic_stream, 0.2, 119.5, 120.0, model=model)
    assert model.seen_samples == pytest.approx(120 * 16000, abs=1600)


def test_no_model_means_the_caption_model_for_the_second_pass(synthetic_stream, monkeypatch):
    fake = Listening([])
    monkeypatch.setattr(transcribe, "caption_model", lambda: fake)
    transcribe.transcribe_clip(synthetic_stream, 30.0, 40.0, 120.0)
    assert fake.seen_beam == 5
