"""One hot spot's window transcribed with word timings, on a plain processor.

Only the window's sound is read (`audio.read_pcm`), so the whole stream is never transcribed.
The model hears window seconds; every time written out is in stream seconds.
"""

from __future__ import annotations

import os
import threading
import unicodedata
from pathlib import Path

from clipper.models import Segment, Transcript, Word
from streamclip import settings
from streamclip.audio import read_pcm

MODEL = settings.WINDOW_MODEL   # the first pass's model, under its older name
RATE = 16000                    # what Whisper listens at
NON_LATIN_LIMIT = 0.30          # captions are off when more of the letters than this are not Latin
# More threads than this make the model slower, not faster, where some processors are the
# slow kind: on a 14-processor Mac, 37 s of speech with `small` took 4.0 s on 8 threads and
# 15.5 s on 14 (2026-11-23). Starwish allows 4, so nothing changes there.
MOST_THREADS = 8
# The model can hear one word as two, "long" then "-term", or "I" then "'m". A word that
# starts with one of these is the tail of the word before it.
JOINERS = ("-", "'", "\u2019")

_models: dict[str, object] = {}
_model_lock = threading.Lock()


def _load(name: str):
    """A faster-whisper model, built once per process (the first call downloads it)."""
    with _model_lock:
        if name not in _models:
            from faster_whisper import WhisperModel
            _models[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                         cpu_threads=min(settings.processors(), MOST_THREADS))
        return _models[name]


def default_model():
    """The first pass's model: fast, for the windows the judge reads."""
    return _load(settings.WINDOW_MODEL)


def caption_model():
    """The second pass's model: accurate, for the stretches that become clips."""
    return _load(settings.CAPTION_MODEL)


def ensure_model() -> None:
    """Fetch both speech models now, so no job has to (used at install time)."""
    default_model()
    caption_model()


def transcribe_window(audio: Path, start: float, end: float, language: str | None = None,
                      model=None, beam_size: int | None = None) -> tuple[Transcript, str]:
    """The words spoken in [start, end) of the stream, and the language they were in.

    `language=None` lets the model detect it. A window with no speech gives an empty
    transcript, which is a normal answer: in gameplay the joke is often on screen.

    The model is always handed samples read by ffmpeg, never a file: faster-whisper's own
    file decoder breaks with the PyAV that Starwish installs."""
    if model is None:
        model = default_model()
    pcm = read_pcm(Path(audio), start, max(0.0, end - start), rate=RATE)
    segments, info = model.transcribe(pcm, language=language, word_timestamps=True,
                                      vad_filter=True,
                                      beam_size=settings.WINDOW_BEAM if beam_size is None else beam_size)
    words: list[Word] = []
    kept: list[Segment] = []
    for s in segments:
        first = len(words)
        for w in s.words or []:
            text = str(w.word or "").strip()
            if not text or w.start is None or w.end is None:
                continue
            prob = float(getattr(w, "probability", 1.0))
            end_at = round(start + float(w.end), 3)
            if words and text.startswith(JOINERS):
                head = words[-1]
                words[-1] = Word(i=head.i, text=head.text + text, start=head.start,
                                 end=max(head.end, end_at), prob=min(head.prob, prob))
                continue
            words.append(Word(i=len(words), text=text, start=round(start + float(w.start), 3),
                              end=end_at, prob=prob))
        if len(words) > first:
            kept.append(Segment(start=round(start + float(s.start), 3),
                                end=round(start + float(s.end), 3),
                                text=str(s.text or "").strip(), first_word=first,
                                last_word=len(words) - 1))
    found = language or getattr(info, "language", None) or "en"
    return Transcript(words=words, segments=kept, duration=float(end)), found


def transcribe_clip(audio: Path, start: float, end: float, duration: float,
                    language: str | None = None, model=None) -> list[Word]:
    """The second pass: the words of one chosen clip, heard again with the accurate setting.

    The stretch is heard with `CAPTION_PAD` seconds either side, kept inside the stream, so
    a word on the clip's edge is heard whole. Only words that touch [start, end) come back,
    in stream seconds, numbered from 0."""
    if model is None:
        model = caption_model()
    first = max(0.0, start - settings.CAPTION_PAD)
    last = end + settings.CAPTION_PAD
    if duration > 0:
        last = min(last, duration)
    transcript, _ = transcribe_window(audio, first, last, language=language, model=model,
                                      beam_size=settings.CAPTION_BEAM)
    inside = [w for w in transcript.words if w.end > start and w.start < end]
    return [Word(i, w.text, w.start, w.end, w.prob) for i, w in enumerate(inside)]


def _is_latin(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith(("LATIN", "FULLWIDTH LATIN"))


def captions_allowed(text: str) -> bool:
    """Whether the caption font (Latin letters only) can write this text.

    Only letters count: digits, punctuation and emoji are ignored. No letters at all is
    allowed, since there is nothing the font would fail on."""
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return True
    foreign = sum(1 for ch in letters if not _is_latin(ch))
    return foreign / len(letters) <= NON_LATIN_LIMIT
