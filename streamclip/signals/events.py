"""Voice events: shouting, screaming, laughter, cheering and gasps, second by second.

A small pre-trained sound classifier (YAMNet, through MediaPipe) scores every 0.975 s of
the stream. Its music and speech scores are kept too, because music with nobody talking
over it is an intermission, not a moment.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from streamclip.audio import read_pcm
from streamclip.errors import StageError
from streamclip.signals.loudness import seconds

EXCITING = ("Shout", "Yell", "Screaming", "Laughter", "Belly laugh", "Giggle", "Gasp",
            "Cheering", "Applause")
MUSIC = ("Music",)
SPEECH = ("Speech",)

MODEL_FILE = "yamnet.tflite"
# The same file at two addresses: MediaPipe's own model store, then a copy. One address is
# not enough: the copy's host answered a build machine "429 Too Many Requests" (it allows each
# network address so many downloads every five minutes, and build machines share theirs), and
# with nowhere else to ask the build failed.
MODEL_URLS = (
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
    "https://huggingface.co/litert-community/MediaPipe-AudioClassifier-YAMNet/resolve/main/yamnet.tflite",
)
MODEL_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"
WAITS = (3.0, 10.0, 30.0)   # seconds before asking both addresses again; then it gives up

RATE = 16000
WINDOW_SAMPLES = 15600      # the classifier scores 0.975 s at a time, end to end
CHUNK_WINDOWS = 615         # windows classified per read: just under ten minutes of sound

# classify(pcm_16k) -> one label-to-score map per 0.975 s window, in order.
Classify = Callable[[np.ndarray], "list[dict[str, float]]"]


@dataclass
class VoiceTrack:
    """One entry per second of the stream."""
    voice: np.ndarray       # the highest exciting score, 0..1
    music: np.ndarray
    speech: np.ndarray
    label: list[str]        # which exciting label that was, or ""


def models_dir() -> Path:
    return Path(os.environ.get("STREAMCLIP_MODELS") or Path.home() / ".cache" / "streamclip")


def _fetch(url: str) -> bytes:
    """The model from one address. Anything that is not the model is an error: a refusal, a
    download cut short, or some other page sent in its place."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise OSError(f"{len(data)} bytes arrived that are not the model")
    return data


def ensure_model() -> Path:
    """The classifier's model file, downloaded on first use.

    Each address is asked in turn. When none sends the model, both are asked again after a
    wait, `len(WAITS)` more times.
    """
    path = models_dir() / MODEL_FILE
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{MODEL_FILE}.{os.getpid()}.partial")
    failed: Exception | None = None
    for wait in (0.0, *WAITS):
        if wait:
            time.sleep(wait)
        for url in MODEL_URLS:
            try:
                data = _fetch(url)
                partial.write_bytes(data)
                os.replace(partial, path)
                return path
            except (OSError, http.client.HTTPException) as exc:
                partial.unlink(missing_ok=True)
                failed = exc
    raise StageError(f"The sound classifier could not be downloaded: {failed}") from failed


def default_classifier() -> Classify:
    """The real classifier: YAMNet through MediaPipe's audio classifier."""
    from mediapipe.tasks.python import audio             # imported here: it is slow to load
    from mediapipe.tasks.python.components import containers
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        category_allowlist=list(EXCITING + MUSIC + SPEECH))
    classifier = audio.AudioClassifier.create_from_options(options)

    def classify(pcm: np.ndarray) -> list[dict[str, float]]:
        clip = containers.AudioData.create_from_array(np.asarray(pcm, dtype=np.float32), RATE)
        results = sorted(classifier.classify(clip), key=lambda r: r.timestamp_ms)
        return [{c.category_name: float(c.score) for c in r.classifications[0].categories}
                for r in results]

    return classify


def voice_events(audio: Path, duration: float, classify: Classify | None = None) -> VoiceTrack:
    """Score every second of the stream. A second takes the highest score of any window
    that overlaps it."""
    if classify is None:
        classify = default_classifier()
    count = seconds(duration)
    voice, music, speech = np.zeros(count), np.zeros(count), np.zeros(count)
    label = [""] * count
    chunk = CHUNK_WINDOWS * WINDOW_SAMPLES                # a whole number of windows
    for start in range(0, count * RATE, chunk):
        pcm = read_pcm(audio, start=start / RATE,
                       duration=min(chunk / RATE, duration - start / RATE), rate=RATE)
        if len(pcm) == 0:
            break
        for i, scores in enumerate(classify(pcm)):
            first = start + i * WINDOW_SAMPLES            # this window's samples in the stream
            best = max(EXCITING, key=lambda name: scores.get(name, 0.0))
            top = float(scores.get(best, 0.0))
            for s in range(first // RATE, min(count, (first + WINDOW_SAMPLES - 1) // RATE + 1)):
                if top > voice[s]:
                    voice[s], label[s] = top, best
                music[s] = max(music[s], *(scores.get(name, 0.0) for name in MUSIC), 0.0)
                speech[s] = max(speech[s], *(scores.get(name, 0.0) for name in SPEECH), 0.0)
    return VoiceTrack(voice, music, speech, label)
