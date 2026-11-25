"""One real call to the judge. It costs money, so it is deselected unless asked for:

    ANTHROPIC_API_KEY=... uv run pytest tests/stream/test_stream_judge_live.py -m live -q

The job is built from the stand-in stream: real contact sheets, made-up transcripts. One
candidate has an obvious build-up and payoff at the stream's first loud moment. The other is
someone adjusting their settings.
"""

import json
import os
import subprocess

import pytest

from clipper import shell
from streamclip.jobdir import JobDir
from streamclip.judge import call
from streamclip.judge.checks import apply_checks

OBVIOUS = [
    (8.0, "Okay chat, last one alive. Nobody make a sound."),
    (14.0, "He is right behind that door, I can hear him breathing."),
    (20.0, "I'm just gonna peek. Real slow."),
    (26.0, "Nothing. See? Easy."),
    (30.0, "OH MY GOD HE WAS IN THE CEILING!"),
    (32.2, "NO NO NO NO NO!"),
    (35.0, "I'm done. I am actually done. Chat, clip that."),
]
NOTHING = [
    (40.0, "So yeah, I updated my drivers yesterday."),
    (55.0, "Let me check the settings real quick."),
    (70.0, "Um, brightness is at fifty."),
    (95.0, "Okay. That looks about the same."),
    (110.0, "Anyway."),
]
WINDOWS = {1: (0.0, 30.0, 60.0, OBVIOUS), 2: (35.0, 95.0, 120.0, NOTHING)}      # start, peak, end, lines


def _transcript(lines) -> dict:
    words, segments = [], []
    for start, sentence in lines:
        first = len(words)
        for k, text in enumerate(sentence.split()):
            words.append({"i": len(words), "text": text, "start": start + 0.3 * k,
                          "end": start + 0.3 * k + 0.25, "prob": 1.0})
        segments.append({"start": start, "end": words[-1]["end"], "text": sentence,
                         "first_word": first, "last_word": len(words) - 1})
    return {"words": words, "segments": segments, "duration": 120.0}


def _sheet(video, start: float, end: float, dest) -> None:
    """Fifteen frames of the window on one sheet, five across."""
    subprocess.run([shell.ffmpeg(), "-v", "error", "-y", "-ss", str(start), "-t", str(end - start),
                    "-i", str(video), "-vf", f"fps=15/{end - start},scale=320:-2,tile=5x3",
                    "-frames:v", "1", "-q:v", "5", str(dest)], check=True, capture_output=True, timeout=120)


@pytest.fixture
def live_job(synthetic_stream, tmp_path) -> JobDir:
    job = JobDir(tmp_path / "job")
    (job.root / "candidates").mkdir(parents=True)
    job.stream.write_text(json.dumps({
        "source": str(synthetic_stream), "kind": "file", "video_id": "", "title": "Horror night",
        "creator_login": "standin", "creator_name": "Stand In", "duration": 120.0, "started_at": "",
        "width": 1280, "height": 720, "fps": 30.0, "chapters": [], "extras": {}}))
    hotspots = []
    for n, (start, peak, end, lines) in WINDOWS.items():
        hotspots.append({"n": n, "peak": peak, "start": start, "end": end, "score": 0.9, "fired": [
            {"signal": "loudness", "t": peak, "strength": 0.9, "detail": "loudness spike"}]})
        folder = job.candidate(n)
        folder.mkdir()
        (folder / "transcript.json").write_text(json.dumps(_transcript(lines)))
        _sheet(synthetic_stream, start, end, folder / "sheet.jpg")
    job.hotspots.write_text(json.dumps({"wanted": 2, "hotspots": hotspots}))
    job.candidates_index.write_text(json.dumps(
        {"language": "en", "captions": True, "ready": [1, 2], "notes": []}))
    return job


@pytest.mark.live
def test_the_real_judge_ranks_the_obvious_moment_first(live_job):
    blocks = call.build_request(live_job.root)["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "text", "image", "text", "image", "text"]
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        pytest.skip("ANTHROPIC_API_KEY is not set")

    raw = call.judge(live_job.root, key)

    assert [c["n"] for c in raw["candidates"]] == [1, 2] and raw["model"]
    picks = apply_checks(raw, live_job, 2)
    assert picks and picks[0].candidate == 1 and picks[0].rank == 1
    best = picks[0]
    assert 0.0 <= best.start < 30.0 < best.end <= 60.0              # the scream is inside it
    assert 15.0 <= (best.end - best.start) / 1.3 <= 40.0
    assert best.title and 1 <= best.score <= 10
