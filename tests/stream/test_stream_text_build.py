"""All the text for one clip: captions.ass, title.png and tag.png."""

from pathlib import Path

import numpy as np
import pytest

from clipper.models import Word
from clipper.pacing import TimeMap
from streamclip.schema import Pick, StreamInfo
from streamclip.text import build_text

RATE = 16000


@pytest.fixture
def pick():
    return Pick(rank=1, candidate=3, start=600.0, end=640.0, score=8, reason="a reason",
                title="He actually did it", caption="", hashtags=[], beats=[], protect=[], weak=False)


@pytest.fixture
def ws():
    return [Word(0, "before", 598.0, 598.5),                 # outside the pick
            Word(1, "so", 601.0, 601.4), Word(2, "there", 601.5, 601.9),
            Word(3, "wait", 612.0, 612.4), Word(4, "what", 612.5, 612.9),
            Word(5, "after", 641.0, 641.5)]                  # outside the pick


@pytest.fixture
def pcm():
    t = np.arange(40 * RATE) / RATE
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _info(kind, name, login):
    return StreamInfo(source="x", kind=kind, video_id="1", title="A stream", creator_login=login,
                      creator_name=name, duration=3600.0, started_at="", width=1280, height=720,
                      fps=30.0, chapters=[], extras={})


@pytest.fixture
def info():
    return _info("twitch", "Some Streamer", "somestreamer")


@pytest.fixture
def file_info():
    return _info("file", "", "")


def test_writes_the_three_files_and_lists_them(tmp_path, pick, ws, pcm, info):
    assets = build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), info, None, True, 1.3)
    assert Path(assets.ass).name == "captions.ass" and Path(assets.ass).exists()
    assert [Path(o.path).name for o in assets.overlays] == ["title.png", "tag.png"]
    assert all(Path(p).is_absolute() for p in [assets.ass] + [o.path for o in assets.overlays])


def test_only_the_words_inside_the_pick_are_captioned(tmp_path, pick, ws, pcm, info):
    text = Path(build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), info, None, True, 1.3).ass).read_text()
    assert "WAIT" in text and "BEFORE" not in text and "AFTER" not in text


def test_captions_follow_the_cuts(tmp_path, pick, ws, pcm, info):
    # a word at source-clip 12.0 s, with 2 s cut before it, shows at cut time 10.0 s
    assets = build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 5.0), (7.0, 40.0)]), info, None, True, 1.3)
    assert "0:00:10.00" in Path(assets.ass).read_text()


def test_captions_off_still_makes_the_stickers(tmp_path, pick, ws, pcm, info):
    assets = build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), info, None, False, 1.3)
    assert assets.ass is None and len(assets.overlays) == 2


def test_no_words_means_no_caption_file(tmp_path, pick, pcm, info):   # Review Focus 1
    assert build_text(tmp_path, pick, [], pcm, 16000, TimeMap([(0.0, 40.0)]), info, None, True, 1.3).ass is None


def test_strong_words_are_starred_in_the_file(tmp_path, pick, pcm, info):
    ws = [Word(0, "fuck", 600.0, 600.4), Word(1, "yes", 600.5, 600.9)]
    assert "F***" in Path(build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), info, None, True, 1.3).ass).read_text()


def test_a_file_with_no_creator_has_no_tag(tmp_path, pick, ws, pcm, file_info):
    assets = build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), file_info, None, True, 1.3)
    assert [Path(o.path).name for o in assets.overlays] == ["title.png"]


def test_the_title_stays_up_three_finished_seconds(tmp_path, pick, ws, pcm, info):
    assets = build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), info, None, True, 1.3)
    assert assets.overlays[0].end == pytest.approx(3.9)


def test_captions_drop_sentence_punctuation_and_empty_words(tmp_path, pick, pcm, info):
    ws = [Word(0, "the", 600.0, 600.3), Word(1, "moon.", 600.4, 600.9), Word(2, "...", 601.0, 601.2),
          Word(3, "really?", 601.3, 601.8), Word(4, "systems,", 601.9, 602.4)]
    events = Path(build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), info, None,
                             True, 1.3).ass).read_text().split("[Events]")[1]
    assert "MOON" in events and "MOON." not in events and "SYSTEMS" in events
    assert "SYSTEMS," not in events and "REALLY?" in events and "..." not in events


def test_the_words_handed_in_keep_their_punctuation(tmp_path, pick, pcm, info):
    ws = [Word(0, "the", 600.0, 600.3), Word(1, "moon.", 600.4, 600.9)]
    build_text(tmp_path, pick, ws, pcm, 16000, TimeMap([(0.0, 40.0)]), info, None, True, 1.3)
    assert ws[1].text == "moon."
