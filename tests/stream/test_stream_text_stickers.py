"""The title sticker and the creator tag: PNG images laid over the video."""

import json
from pathlib import Path

import pytest
from PIL import Image

from streamclip.text import stickers


def tag_text(overlay):
    return json.loads(Path(overlay.path + ".json").read_text())["text"]


@pytest.fixture
def avatar_file(tmp_path):
    path = tmp_path / "avatar.jpg"
    Image.new("RGB", (300, 200), (200, 40, 90)).save(path)
    return path


def test_clean_keeps_only_what_the_font_can_draw():                  # Review Focus 4
    assert stickers.clean("Kai 🔥 Cenat") == "Kai Cenat"
    assert stickers.clean("日本語") == "" and stickers.clean("Señor  Müller") == "Señor Müller"


def test_clean_drops_control_characters():
    assert stickers.clean("a\tb\x85c\x7fd\x00") == "a b cd"


def test_title_sticker_is_placed_under_the_seam(tmp_path):
    o = stickers.title_sticker("He actually did it", tmp_path / "t.png", 1.3)
    w, h = Image.open(o.path).size
    assert o.y == 732 and abs((o.x + w / 2) - 540) <= 1 and w <= 820 and o.x + w <= 950
    assert (o.start, o.end) == (0.0, pytest.approx(3.9))             # 3 finished seconds, in cut time


def test_title_uses_the_accent_with_black_letters(tmp_path):
    im = Image.open(stickers.title_sticker("Short one", tmp_path / "t.png", 1.3).path).convert("RGBA")
    colours = {c for _, c in im.getcolors(maxcolors=1 << 20)}
    assert (0xD7, 0xFF, 0x3E, 255) in colours and (0, 0, 0, 255) in colours
    assert im.getpixel((0, 0))[3] == 0                               # see-through corners: it is slanted


def test_a_short_title_stays_big_and_on_one_line(tmp_path):
    a = Image.open(stickers.title_sticker("Short one", tmp_path / "a.png", 1.3).path).size
    b = Image.open(stickers.title_sticker("Short one and then some more words", tmp_path / "b.png", 1.3).path).size
    assert a[1] < b[1]                                               # the longer one wraps


def test_a_long_title_wraps_to_two_lines_then_shrinks_then_cuts(tmp_path):
    o = stickers.title_sticker("this is a very long title that keeps going well past what fits here", tmp_path / "t.png", 1.3)
    w, h = Image.open(o.path).size
    assert w <= 820 and h <= 330
    assert tag_text(o).endswith("...")


def test_one_enormous_word_still_fits(tmp_path):
    o = stickers.title_sticker("A" * 200, tmp_path / "t.png", 1.3)
    assert Image.open(o.path).size[0] <= 820


def test_a_title_with_nothing_drawable_gives_no_sticker(tmp_path):
    assert stickers.title_sticker("🔥🔥🔥", tmp_path / "t.png", 1.3) is None


def test_tag_sits_above_the_seam_on_the_left(tmp_path):
    o = stickers.creator_tag("Some Streamer", "somestreamer", None, tmp_path / "g.png")
    w, h = Image.open(o.path).size
    assert o.x == 40 and o.y + h == 668 and w <= 560 and o.y >= 192 and (o.start, o.end) == (0.0, None)


def test_tag_is_a_slanted_black_pill_with_white_letters(tmp_path):
    im = Image.open(stickers.creator_tag("Name", "name", None, tmp_path / "g.png").path).convert("RGBA")
    colours = {c for _, c in im.getcolors(maxcolors=1 << 20)}
    assert (0, 0, 0, 255) in colours and (255, 255, 255, 255) in colours
    assert (0xD7, 0xFF, 0x3E, 255) in colours and im.getpixel((0, 0))[3] == 0
    assert tag_text(stickers.creator_tag("Name", "name", None, tmp_path / "g.png")) == "@NAME"


def test_tag_with_a_picture_is_wider_than_without(tmp_path, avatar_file):
    a = Image.open(stickers.creator_tag("Name", "name", avatar_file, tmp_path / "a.png").path).size
    b = Image.open(stickers.creator_tag("Name", "name", None, tmp_path / "b.png").path).size
    assert a[0] > b[0]


def test_a_broken_picture_falls_back_to_the_name_alone(tmp_path):
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"not an image")
    assert stickers.creator_tag("Name", "name", bad, tmp_path / "g.png") is not None


def test_a_missing_picture_falls_back_to_the_name_alone(tmp_path):
    assert stickers.creator_tag("Name", "name", tmp_path / "nope.jpg", tmp_path / "g.png") is not None


def test_an_undrawable_name_falls_back_to_the_login(tmp_path):        # Review Focus 4
    assert tag_text(stickers.creator_tag("日本語の名前", "somestreamer", None, tmp_path / "g.png")) == "@SOMESTREAMER"


def test_no_name_at_all_gives_no_tag(tmp_path):
    assert stickers.creator_tag("", "", None, tmp_path / "g.png") is None


def test_a_very_long_name_is_cut_to_fit(tmp_path):
    assert Image.open(stickers.creator_tag("x" * 80, "x", None, tmp_path / "g.png").path).size[0] <= 560
