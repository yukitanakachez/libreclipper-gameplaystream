"""Fakes and ready-made job folders the stage tests share. Not a test file itself.

Nothing here downloads anything: the sound classifier hears nothing, the speech model says the
same steady words in every window, and the face detector sees one small face inside the test
stream's camera box. A job folder is taken through each stage once per test run and copied for
every test that needs one, because a stage takes seconds and a copy takes none.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper.jobs import read_json, write_json
from clipper.models import Rect
from streamclip import stages
from streamclip.jobdir import JobDir

PICK1_CAPTION = "He did not see that coming."
BURSTS = (31.0, 96.0)               # the middle of the test stream's two loud moments
FACE = Rect(0.11, 0.08, 0.06, 0.12)  # inside the camera box (24,24 320x180 of 1280x720)
RATE = 16000


class FakeSpeech:
    """A word every 0.6 s for the whole window, a sentence every five words."""

    def __init__(self, vocabulary=("oh", "no", "not", "like", "this."), language="en"):
        self.vocabulary, self.language = vocabulary, language

    def transcribe(self, pcm, language=None, word_timestamps=False, vad_filter=False,
                   beam_size=None, **_):
        self.beams = [*getattr(self, "beams", []), beam_size]
        count = int(len(pcm) / RATE / 0.6)
        words = [SimpleNamespace(word=" " + self.vocabulary[i % len(self.vocabulary)],
                                 start=i * 0.6, end=i * 0.6 + 0.5, probability=0.9)
                 for i in range(count)]
        segments = [SimpleNamespace(start=chunk[0].start, end=chunk[-1].end,
                                    text="".join(w.word for w in chunk).strip(), words=chunk)
                    for chunk in (words[i:i + 5] for i in range(0, len(words), 5))]
        return iter(segments), SimpleNamespace(language=language or self.language)


def hears_nothing(pcm):
    return [{} for _ in range(len(pcm) // 15600 + 1)]


def small_face(video, start, end, fps=10.0, **how):
    return [(start + i / fps, FACE, 0.0) for i in range(int((end - start) * fps))]


SECOND_LISTEN = ("yes", "we", "did", "it", "now.")      # what the accurate pass hears instead


def install_fakes(monkeypatch, speech=None, again=None):
    speech = speech or FakeSpeech()
    again = again or FakeSpeech(SECOND_LISTEN)
    monkeypatch.setattr("streamclip.signals.events.default_classifier", lambda: hears_nothing)
    monkeypatch.setattr("streamclip.windows.transcribe.default_model", lambda: speech)
    monkeypatch.setattr("streamclip.windows.transcribe.caption_model", lambda: again)
    monkeypatch.setattr("streamclip.framing.smallfaces.detect_faces", small_face)
    for name in ("JUDGE_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fakes(monkeypatch):
    install_fakes(monkeypatch)


@pytest.fixture
def job(tmp_path):
    return JobDir(tmp_path / "job")


def good_raw(job: JobDir) -> dict:
    """What a judge would say: keep the two loud moments, the first one best."""
    rows = []
    for spot in read_json(job.hotspots)["hotspots"]:
        for k, burst in enumerate(BURSTS):
            if any(row["burst"] == burst for row in rows):
                continue
            if spot["start"] <= burst - 19 and burst + 5 <= spot["end"]:
                at = burst - spot["start"]              # the judge answers in window seconds
                rows.append({
                    "burst": burst, "n": spot["n"], "keep": True, "score": 9 - 2 * k,
                    "reason": "A loud payoff after a steady build-up.",
                    "start": at - 19, "end": at + 4,
                    "title": ("The first loud one", "Then it happened again")[k],
                    "caption": (PICK1_CAPTION, "Twice in one stream.")[k],
                    "hashtags": ["gaming", "#funny"],
                    "beats": [{"kind": "payoff", "t": at - 1}, {"kind": "reaction", "t": at + 1}],
                    "protect": []})
                break
    assert len(rows) == 2, "each loud moment needs a hot spot of its own"
    return {"candidates": [{k: v for k, v in row.items() if k != "burst"} for row in rows]}


_READY: dict[str, Path] = {}


def _template(name: str, tmp_path_factory, build) -> Path:
    if name not in _READY:
        root = tmp_path_factory.mktemp(name) / "job"
        with pytest.MonkeyPatch.context() as patch:
            install_fakes(patch)
            build(root)
        _READY[name] = root
    return _READY[name]


def found_template(tmp_path_factory, synthetic_stream) -> Path:
    return _template("found", tmp_path_factory,
                     lambda root: stages.find(str(synthetic_stream), JobDir(root), wanted=2,
                                              progress=lambda line: None))


def judged_template(tmp_path_factory, synthetic_stream) -> Path:
    def build(root: Path) -> None:
        shutil.copytree(found_template(tmp_path_factory, synthetic_stream), root)
        job = JobDir(root)
        write_json(root / "raw.json", good_raw(job))
        stages.judge(job, raw=root / "raw.json", progress=lambda line: None)
    return _template("judged", tmp_path_factory, build)


def made_template(tmp_path_factory, synthetic_stream) -> Path:
    def build(root: Path) -> None:
        shutil.copytree(judged_template(tmp_path_factory, synthetic_stream), root)
        stages.make(JobDir(root), progress=lambda line: None)
    return _template("made", tmp_path_factory, build)


def copy_of(template: Path, tmp_path: Path) -> JobDir:
    shutil.copytree(template, tmp_path / "job")
    return JobDir(tmp_path / "job")


@pytest.fixture
def found_job(tmp_path, tmp_path_factory, synthetic_stream, fakes):
    return copy_of(found_template(tmp_path_factory, synthetic_stream), tmp_path)


@pytest.fixture
def judged_job(tmp_path, tmp_path_factory, synthetic_stream, fakes):
    return copy_of(judged_template(tmp_path_factory, synthetic_stream), tmp_path)


@pytest.fixture
def made_job(tmp_path, tmp_path_factory, synthetic_stream, fakes):
    return copy_of(made_template(tmp_path_factory, synthetic_stream), tmp_path)
