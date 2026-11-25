"""The Shout captions: loudness-sized words, the look, the width limit and the starring."""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageFont

from clipper.captions import Chunk
from clipper.models import Word
from streamclip.text import FONTS_DIR, shout
from streamclip.text.censor import censor

FIXTURES = Path(__file__).parent / "fixtures"
RATE = 16000


def words(*triples):                                  # (text, start, end) in stream seconds from 600
    return [Word(i, t, 600 + a, 600 + b) for i, (t, a, b) in enumerate(triples)]


def speech_at(segments, length=None):
    """Clip sound: a 220 Hz tone at the given RMS level (dBFS) in each (start, end, db)."""
    length = length or max(b for _, b, _ in segments) + 0.5
    t = np.arange(int(length * RATE)) / RATE
    pcm = np.zeros_like(t, dtype=np.float32)
    for a, b, db in segments:
        inside = (t >= a) & (t < b)
        pcm[inside] = (np.sqrt(2) * 10 ** (db / 20) * np.sin(2 * np.pi * 220 * t[inside]))
    return pcm


def chunks_of(*triples):
    """One chunk of words already in cut time."""
    ws = [Word(i, t, a, b) for i, (t, a, b) in enumerate(triples)]
    return [Chunk(ws, ws[0].start, ws[-1].end, " ".join(w.text for w in ws))]


def dialogue_lines(text):
    return [line for line in text.splitlines() if line.startswith("Dialogue:")]


def style_line(text):
    return next(line for line in text.splitlines() if line.startswith("Style: Shout,"))


def runs(line):
    """(override tags, text) pairs from a dialogue line's text field."""
    body = line.split(",", 9)[9]
    return re.findall(r"\{([^}]*)\}([^{]*)", body)


def accented_word(line):
    for tags, text in runs(line):
        if "&H3EFFD7&" in tags and text.strip():
            return text.strip()
    return None


@lru_cache(maxsize=None)
def _anton(size):
    return ImageFont.truetype(str(FONTS_DIR / "Anton-Regular.ttf"), size)


def rendered_width(line):
    """Width of the line's letters as libass draws them, measured independently of shout.py.

    libass sizes a font by usWinAscent + usWinDescent (2876 + 674 for Anton, 2048 units/em)."""
    size = 191                                       # 110 px to the em
    total = 0.0
    for tags, text in runs(line):
        if m := re.search(r"\\fs(\d+)", tags):
            size = int(m.group(1))
        scale = int(re.search(r"\\fscx(\d+)", tags).group(1)) if "\\fscx" in tags else 100
        total += _anton(100).getlength(text) / 100 * size * scale / 100 * 2048 / 3550
    return total


GOLDEN_CHUNKS = [
    Chunk([Word(0, "what's", 0.2, 0.6), Word(1, "up?", 0.65, 1.1)], 0.2, 1.1, "what's up?"),
    Chunk([Word(2, "I'm", 1.6, 1.9), Word(3, "so", 1.95, 2.3)], 1.6, 2.3, "I'm so"),
    Chunk([Word(4, "sorry", 2.35, 3.0)], 2.35, 3.0, "sorry"),
    Chunk([Word(5, "get", 3.6, 3.9), Word(6, "out!", 3.95, 4.8)], 3.6, 4.8, "get out!"),
]
GOLDEN_SCALES = {0: 100, 1: 120, 2: 100, 3: 140, 4: 100, 5: 120, 6: 165}


def test_louder_words_get_bigger_in_four_steps():
    ws = words(("so", 0.0, 0.4), ("i", 0.5, 0.7), ("went", 0.8, 1.2), ("in", 1.3, 1.5), ("NO", 2.0, 2.6))
    pcm = speech_at([(0.0, 1.5, -24.0), (2.0, 2.6, -8.0)])       # the last word is 16 dB hotter
    scales = shout.word_scales(ws, pcm, 16000, 600.0)
    assert set(scales.values()) <= {100, 120, 140, 165} and scales[4] == 165 and scales[0] == 100


def test_each_step_has_its_threshold():
    ws = words(*[(c, i * 0.5, i * 0.5 + 0.4) for i, c in enumerate("abcdefg")])
    levels = [-30, -30, -30, -30, -23.5, -20.5, -17.5]          # +0, +6.5, +9.5, +12.5 over the median
    pcm = speech_at([(i * 0.5, i * 0.5 + 0.4, db) for i, db in enumerate(levels)])
    scales = shout.word_scales(ws, pcm, 16000, 600.0)
    assert [scales[i] for i in range(7)] == [100, 100, 100, 100, 120, 140, 165]


def test_a_clip_of_one_level_stays_at_normal_size():
    ws = words(("a", 0.0, 0.4), ("b", 0.5, 0.9), ("c", 1.0, 1.4))
    assert set(shout.word_scales(ws, speech_at([(0.0, 1.4, -20.0)]), 16000, 600.0).values()) == {100}


def test_two_words_or_fewer_are_never_scaled():
    assert shout.word_scales(words(("oh", 0.0, 0.4)), speech_at([(0.0, 0.4, -3.0)]), 16000, 600.0) == {0: 100}


def test_words_past_the_end_of_the_sound_stay_normal():
    ws = words(("a", 0.0, 0.4), ("b", 0.5, 0.9), ("c", 9.0, 9.4))
    scales = shout.word_scales(ws, speech_at([(0.0, 0.9, -20.0)]), 16000, 600.0)
    assert scales == {0: 100, 1: 100, 2: 100}


def test_ass_has_the_shout_look():
    text = shout.build_shout_ass(chunks_of(("oh", 0.0, 0.3), ("no", 0.35, 0.8)), {0: 100, 1: 165}, 5.0)
    assert "PlayResX: 1080" in text and "PlayResY: 1920" in text
    assert "Style: Shout,Anton," in text and r"\pos(540,1363)" in text and r"\frz4" in text
    assert "OH" in text and "NO" in text and "oh" not in text.split("[Events]")[1]          # capitals
    assert r"\fscx165\fscy165" in text and "&H3EFFD7&" in text                              # the accent, as BGR


def test_a_shouted_word_shakes_and_turns_the_chunk_accent():
    events = dialogue_lines(shout.build_shout_ass(
        chunks_of(("get", 0.0, 0.3), ("out", 0.35, 0.8)), {0: 100, 1: 165}, 5.0))
    assert all("&HFFFFFF&" not in tags for line in events for tags, _ in runs(line))
    assert r"\t(0,60,\frz8)\t(60,120,\frz1)\t(120,180,\frz4)" in events[1]
    assert r"\t(" not in events[0]


def test_the_spoken_word_is_the_accented_one():
    events = dialogue_lines(shout.build_shout_ass(chunks_of(("oh", 0.0, 0.3), ("no", 0.35, 0.8)), {0: 100, 1: 100}, 5.0))
    assert len(events) == 2 and accented_word(events[0]) == "OH" and accented_word(events[1]) == "NO"


def test_words_hold_until_the_next_one_and_stop_at_the_end():
    events = dialogue_lines(shout.build_shout_ass(chunks_of(("oh", 0.0, 0.3), ("no", 0.35, 9.0)), {}, 5.0))
    assert events[0].split(",")[1:3] == ["0:00:00.00", "0:00:00.35"]
    assert events[1].split(",")[1:3] == ["0:00:00.35", "0:00:05.00"]


def test_at_most_two_words_show_at_once():
    ws = words(*[(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate("this is a much longer sentence".split())])
    assert max(len(c.words) for c in shout.chunk(ws)) == 2


def test_nothing_is_wider_than_820_pixels():
    long = chunks_of(("unbelievable", 0.0, 0.5), ("congratulations", 0.6, 1.2))
    for line in dialogue_lines(shout.build_shout_ass(long, {0: 165, 1: 165}, 5.0)):
        assert rendered_width(line) <= 820


def test_a_wide_line_gives_up_its_scale_before_its_size():
    events = dialogue_lines(shout.build_shout_ass(chunks_of(("get", 0.0, 0.3), ("wrecked", 0.35, 0.8)),
                                                  {0: 100, 1: 165}, 5.0))
    assert r"\fs" not in events[0].replace(r"\fscx", "").replace(r"\fscy", "")
    assert 100 < int(re.findall(r"\\fscx(\d+)", events[0])[1]) < 165
    assert rendered_width(events[0]) <= 820


def test_text_width_grows_with_the_scale():
    assert shout.text_width("NO", 165) == pytest.approx(shout.text_width("NO") * 1.65)
    assert 0 < shout.text_width("I") < shout.text_width("WWW")


def test_no_words_gives_a_valid_file_with_no_lines():                # Review Focus 1
    text = shout.build_shout_ass([], {}, 10.0)
    assert "[Events]" in text and "Dialogue:" not in text


def test_the_look_is_not_the_reference_clips():
    text = shout.build_shout_ass(chunks_of(("oh", 0.0, 0.3)), {0: 100}, 5.0)
    assert "BorderStyle=3" not in text and ",3," not in style_line(text)       # no box behind the words


def test_braces_in_words_cannot_open_override_tags():
    text = shout.build_shout_ass(chunks_of(("{\\b1}x", 0.0, 0.3)), {}, 5.0)
    assert r"\{\\B1\}X" in text


def test_golden_file():
    assert shout.build_shout_ass(GOLDEN_CHUNKS, GOLDEN_SCALES, 6.0) == (FIXTURES / "golden_shout.ass").read_text()


@pytest.mark.parametrize("word, shown", [
    ("fuck", "f***"), ("Fucking", "F******"), ("shit!", "s***!"), ("damn", "damn"), ("class", "class"), ("bitch.", "b****."),
    ("MOTHERFUCKER", "M***********"), ("assholes,", "a*******,"), ("\"cunt\"", "\"c***\""), ("dickens", "dickens"),
    ("", "")])
def test_strong_words_are_starred(word, shown):
    assert censor(word) == shown


@pytest.mark.parametrize("word, shown", [
    ("moon.", "moon"), ("systems.", "systems"), ("so,", "so"), ("wait;", "wait"), ("look:", "look"),
    ("what?", "what?"), ("no!", "no!"), ("what?!", "what?!"), ("don't", "don't"), ("it's.", "it's"),
    ("u.s.", "u.s"), ("yes...", "yes"), ("…", ""), (".", ""), ("long-term,", "long-term")])
def test_caption_words_lose_sentence_punctuation(word, shown):
    assert shout.caption_word(word) == shown
