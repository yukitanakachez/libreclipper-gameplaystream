"""Lane D, task 3: every hot spot turned into a candidate folder the judge can read."""

import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from clipper import shell
from clipper.jobs import read_json
from clipper.models import Transcript
from streamclip.errors import NoVideo, StageError
from streamclip.jobdir import JobDir
from streamclip.schema import CandidateIndex, Hotspot
from streamclip.windows import build, build_candidates


def hot(n, peak):                       # a Hotspot whose window is clamped to the 120 s fixture
    return Hotspot(n, peak, max(0.0, peak - 60), min(120.0, peak + 30), 0.9, [])


def seg(start, end, words):
    return SimpleNamespace(
        start=start, end=end, text=" ".join(w for w, _, _ in words),
        words=[SimpleNamespace(word=" " + w, start=a, end=b, probability=0.9) for w, a, b in words])


ENGLISH = [seg(1.0, 2.0, [("oh", 1.0, 1.3), ("no!", 1.4, 2.0)])]
JAPANESE = [seg(1.0, 2.0, [("これは", 1.0, 1.3), ("何ですか", 1.4, 2.0)])]


class FakeModel:
    def __init__(self, segments, language):
        self.segments, self.language = segments, language
        self.languages_asked = []
        self.threads = set()

    def transcribe(self, pcm, language=None, word_timestamps=False, vad_filter=False, **_):
        self.languages_asked.append(language)
        self.threads.add(threading.get_ident())
        return iter(self.segments), SimpleNamespace(language=language or self.language)


class FakeSource:
    """Cuts windows out of the synthetic stream, and can be told to fail some of them."""

    def __init__(self, stream: Path, cache: Path):
        self.stream, self.cache = stream, cache
        self.failing: list[float] = []
        self.fetched: list[tuple[float, float, int]] = []
        self.lock = threading.Lock()

    def fail_video_for(self, peak: float) -> None:
        self.failing.append(peak)

    def fetch_video(self, start, end, max_height, dest):
        with self.lock:
            self.fetched.append((start, end, max_height))
        if any(start <= p < end for p in self.failing):
            raise NoVideo("the stretch came back with sound only")
        cut = self.cache / f"{start}-{end}-{max_height}.mp4"      # each cut encoded once
        with self.lock:
            if not cut.exists():
                shell.run([shell.ffmpeg(), "-v", "error", "-y", "-ss", f"{start}",
                           "-t", f"{end - start}", "-i", str(self.stream),
                           "-vf", f"scale=-2:{max_height}", "-c:v", "libx264",
                           "-preset", "ultrafast", "-c:a", "aac", str(cut)])
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cut, dest)
        return Path(dest)


@pytest.fixture
def job(tmp_path, synthetic_stream):
    job = JobDir(tmp_path / "job")
    job.audio.parent.mkdir(parents=True)
    os.symlink(synthetic_stream, job.audio)            # ffmpeg reads the mp4 for its sound
    return job


@pytest.fixture(scope="module")
def window_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("windows")


@pytest.fixture
def fake_source(synthetic_stream, window_cache):
    return FakeSource(synthetic_stream, window_cache)


@pytest.fixture
def fake_model():
    return FakeModel(ENGLISH, "en")


def test_each_hot_spot_gets_its_three_files(job, fake_source, fake_model):
    index = build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)], model=fake_model)
    for n in (1, 2):
        d = job.candidate(n)
        assert (d / "transcript.json").exists() and (d / "window.mp4").exists() and (d / "sheet.jpg").exists()
    assert index.ready == [1, 2] and index.language == "en" and index.captions is True
    assert read_json(job.candidates_index)["ready"] == [1, 2]
    assert CandidateIndex.from_dict(read_json(job.candidates_index)) == index


def test_windows_are_fetched_small_and_from_the_hot_spot(job, fake_source, fake_model):
    build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)], model=fake_model)
    assert sorted(fake_source.fetched) == [(0.0, 61.0, 360), (36.0, 120.0, 360)]
    probe = shell.probe(job.candidate(2) / "window.mp4")
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video["height"] == 360


def test_transcripts_are_in_stream_seconds(job, fake_source, fake_model):
    build_candidates(job, fake_source, [hot(2, 96.0)], model=fake_model)
    t = Transcript.from_dict(read_json(job.candidate(2) / "transcript.json"))
    assert [(w.text, w.start) for w in t.words] == [("oh", 37.0), ("no!", 37.4)]
    assert t.duration == 120.0


def test_a_failed_video_fetch_skips_that_candidate_only(job, fake_source, fake_model):
    fake_source.fail_video_for(96.0)                       # raises NoVideo for that window
    index = build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)], model=fake_model)
    assert index.ready == [1] and "candidate 02" in index.notes[0]
    assert "sound only" in index.notes[0]


def test_stale_files_from_an_earlier_run_do_not_make_a_candidate_ready(job, fake_source, fake_model):
    d = job.candidate(2)
    d.mkdir(parents=True)
    for name in ("transcript.json", "window.mp4", "sheet.jpg"):
        (d / name).write_text("old")
    fake_source.fail_video_for(96.0)
    index = build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)], model=fake_model)
    assert index.ready == [1] and not (d / "sheet.jpg").exists()


def test_no_usable_candidate_is_a_stage_error(job, fake_source, fake_model):
    fake_source.fail_video_for(31.0)
    with pytest.raises(StageError, match="None of the hot spots"):
        build_candidates(job, fake_source, [hot(1, 31.0)], model=fake_model)


def test_no_hot_spots_is_a_stage_error(job, fake_source, fake_model):
    with pytest.raises(StageError, match="None of the hot spots"):
        build_candidates(job, fake_source, [], model=fake_model)


def test_a_failed_transcription_skips_that_candidate_only(job, fake_source):
    class Flaky(FakeModel):
        def transcribe(self, pcm, **kw):
            if len(self.languages_asked) == 0:
                self.languages_asked.append("boom")
                raise RuntimeError("the model ran out of memory")
            return super().transcribe(pcm, **kw)

    index = build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)],
                             model=Flaky(ENGLISH, "en"))
    assert index.ready == [2] and index.notes == ["candidate 01 skipped: the model ran out of memory"]


def test_the_language_is_detected_once_then_reused(job, fake_source, fake_model):
    build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)], model=fake_model)
    assert fake_model.languages_asked == [None, "en"]


def test_a_silent_first_window_does_not_decide_the_language(job, fake_source):
    """Detection on a window with no words is a guess, so the next window detects again."""
    class FirstSilent(FakeModel):
        def transcribe(self, pcm, language=None, **kw):
            segs, info = super().transcribe(pcm, language=language, **kw)
            return (iter([]) if len(self.languages_asked) == 1 else segs), info

    model = FirstSilent(ENGLISH, "en")
    index = build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0), hot(3, 110.0)],
                             model=model)
    assert model.languages_asked == [None, None, "en"] and index.ready == [1, 2, 3]


def test_a_window_with_no_speech_is_still_ready(job, fake_source):       # Review Focus 1
    index = build_candidates(job, fake_source, [hot(1, 75.0)], model=FakeModel([], "en"))
    assert index.ready == [1] and index.captions is True and index.language == "en"
    assert read_json(job.candidate(1) / "transcript.json")["words"] == []


def test_a_non_latin_stream_turns_captions_off(job, fake_source):
    index = build_candidates(job, fake_source, [hot(1, 31.0)], model=FakeModel(JAPANESE, "ja"))
    assert index.captions is False and index.language == "ja"


def test_transcription_runs_in_the_calling_thread(job, fake_source, fake_model):
    build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)], model=fake_model)
    assert fake_model.threads == {threading.get_ident()}


def test_progress_counts_candidates(job, fake_source, fake_model):
    lines = []
    build_candidates(job, fake_source, [hot(1, 31.0), hot(2, 96.0)], progress=lines.append, model=fake_model)
    assert lines[-1] == "Windows: 2 of 2"
    assert "Windows: 1 of 2" in lines


def test_no_model_means_the_default_model(job, fake_source, monkeypatch):
    fake = FakeModel(ENGLISH, "en")
    monkeypatch.setattr(build.transcribe, "default_model", lambda: fake)
    build_candidates(job, fake_source, [hot(1, 31.0)])
    assert fake.languages_asked == [None]


def test_the_sheet_is_made_from_the_window(job, fake_source, fake_model):
    build_candidates(job, fake_source, [hot(1, 31.0)], model=fake_model)
    with Image.open(job.candidate(1) / "sheet.jpg") as im:
        assert im.size == (1920, 648)
