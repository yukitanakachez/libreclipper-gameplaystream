"""The title sticker and the creator tag, drawn with Pillow as see-through PNGs.

Title: black Anton capitals on an accent rectangle with a hard black shadow, slanted 3 degrees
anticlockwise, under the seam for the first three finished seconds. Tag: a black pill with an
accent edge, the creator's picture and `@NAME`, slanted the same way, just above the seam.
Each PNG gets a small JSON note beside it (`<name>.png.json`) saying what it reads.
"""

from __future__ import annotations

import functools
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from streamclip.schema import Overlay
from streamclip.settings import ACCENT, FACE_H, OUT_W, TITLE_SECONDS
from streamclip.text import FONT

SLANT = 3                                   # degrees, anticlockwise
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
ACCENT_RGBA = (int(ACCENT[1:3], 16), int(ACCENT[3:5], 16), int(ACCENT[5:7], 16), 255)

TITLE_TOP = FACE_H + 40                     # 732: 40 px under the seam
TITLE_MAX_W = 820
TITLE_SIZES = range(96, 55, -4)             # 96 down to 56
TITLE_PAD = 28
TITLE_SHADOW = 10
TITLE_LEADING = 0.18                        # gap between the two lines, as a share of the size

TAG_LEFT, TAG_BOTTOM = 40, FACE_H - 24      # 40, 668: 24 px above the seam
TAG_MAX_W = 560
TAG_SIZE = 44
TAG_EDGE = 4
TAG_PICTURE = 64
TAG_INSET = 8                               # around the picture
TAG_PAD = 28                                # either side of the name
ELLIPSIS = "..."


def clean(text: str) -> str:
    """Only what Anton can draw: U+0020 to U+017F without the control characters, spaces
    squeezed."""
    kept = "".join(c if 0x20 <= ord(c) <= 0x17F and not 0x7F <= ord(c) <= 0x9F
                   else (" " if c.isspace() else "") for c in text)
    return re.sub(r"\s+", " ", kept).strip()


@functools.lru_cache(maxsize=None)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def _upper(text: str) -> str:
    return clean(text.upper())              # some capitals fall outside the range ('ŉ' -> 'ʼN')


def _slant(image: Image.Image) -> Image.Image:
    return image.rotate(SLANT, resample=Image.BICUBIC, expand=True)


def _note(dest: Path, text: str) -> None:
    dest.with_name(dest.name + ".json").write_text(json.dumps({"text": text}))


# ---------------------------------------------------------------- the title sticker

def _title_box(lines: list[str], size: int) -> tuple[int, int, int, int]:
    """(box width, box height, cap height, line gap) before the shadow and the slant."""
    font = _font(size)
    cap = font.getbbox("H", anchor="ls")[1] * -1           # cap height above the baseline
    gap = round(size * TITLE_LEADING)
    box_w = round(max(font.getlength(line) for line in lines)) + 2 * TITLE_PAD
    box_h = len(lines) * cap + (len(lines) - 1) * gap + 2 * TITLE_PAD
    return box_w, box_h, cap, gap


def _slanted_width(w: int, h: int) -> float:
    a = math.radians(SLANT)
    return w * math.cos(a) + h * math.sin(a) + 2           # +2: Pillow rounds the canvas up


def _title_fits(lines: list[str], size: int) -> bool:
    box_w, box_h, _, _ = _title_box(lines, size)
    return _slanted_width(box_w + TITLE_SHADOW, box_h + TITLE_SHADOW) <= TITLE_MAX_W


def _title_image(lines: list[str], size: int) -> Image.Image:
    font = _font(size)
    box_w, box_h, cap, gap = _title_box(lines, size)
    canvas = Image.new("RGBA", (box_w + TITLE_SHADOW, box_h + TITLE_SHADOW), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([TITLE_SHADOW, TITLE_SHADOW, box_w + TITLE_SHADOW - 1, box_h + TITLE_SHADOW - 1],
                   fill=BLACK)
    draw.rectangle([0, 0, box_w - 1, box_h - 1], fill=ACCENT_RGBA)
    for k, line in enumerate(lines):
        baseline = TITLE_PAD + cap + k * (cap + gap)
        draw.text(((box_w - font.getlength(line)) / 2, baseline), line, font=font, fill=BLACK, anchor="ls")
    return _slant(canvas)


def _layouts(words: list[str]) -> list[list[str]]:
    """One line, then every two-line split, most even first."""
    out = [[" ".join(words)]]
    splits = [[" ".join(words[:i]), " ".join(words[i:])] for i in range(1, len(words))]
    return out + sorted(splits, key=lambda pair: abs(len(pair[0]) - len(pair[1])))


def _fit_title(text: str) -> tuple[list[str], int, str]:
    words = text.split()
    for size in TITLE_SIZES:
        for lines in _layouts(words):
            if _title_fits(lines, size):
                return lines, size, " ".join(lines)
    smallest = TITLE_SIZES[-1]
    while len(words) > 1:                   # cut words from the end
        words = words[:-1]
        shown = " ".join(words) + ELLIPSIS
        for lines in _layouts(shown.split()):
            if _title_fits(lines, smallest):
                return lines, smallest, shown
    word = words[0]                         # one word too long by itself: cut letters
    while len(word) > 1:
        word = word[:-1]
        if _title_fits([word + ELLIPSIS], smallest):
            break
    return [word + ELLIPSIS], smallest, word + ELLIPSIS


def title_sticker(title: str, dest: Path, speed: float) -> Overlay | None:
    """The title sticker, centred under the seam for TITLE_SECONDS finished seconds."""
    text = _upper(title)
    if not text:
        return None
    lines, size, shown = _fit_title(text)
    image = _title_image(lines, size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    _note(dest, shown)
    return Overlay(str(dest), OUT_W // 2 - image.width // 2, TITLE_TOP, 0.0, TITLE_SECONDS * speed)


# ---------------------------------------------------------------- the creator tag

def _picture(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        with Image.open(path) as im:
            im = ImageOps.fit(im.convert("RGB"), (TAG_PICTURE * 4, TAG_PICTURE * 4))
    except Exception:                       # missing, broken or not a picture: name alone
        return None
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, im.width - 1, im.height - 1], fill=255)
    out = Image.new("RGBA", im.size, (0, 0, 0, 0))
    out.paste(im, (0, 0), mask)
    return out.resize((TAG_PICTURE, TAG_PICTURE), Image.LANCZOS)


def _tag_image(text: str, picture: Image.Image | None) -> Image.Image:
    font = _font(TAG_SIZE)
    height = TAG_PICTURE + 2 * TAG_INSET
    left = TAG_INSET + TAG_PICTURE + 16 if picture is not None else TAG_PAD
    width = left + round(font.getlength(text)) + TAG_PAD
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=height // 2, fill=BLACK,
                           outline=ACCENT_RGBA, width=TAG_EDGE)
    if picture is not None:
        canvas.alpha_composite(picture, (TAG_INSET, TAG_INSET))
    draw.text((left, height / 2), text, font=font, fill=WHITE, anchor="lm")
    return _slant(canvas)


def creator_tag(name: str, login: str, avatar: Path | None, dest: Path) -> Overlay | None:
    """The creator tag, from the left edge just above the seam, for the whole clip."""
    shown = _upper(name) or _upper(login)
    if not shown:
        return None
    picture = _picture(avatar)
    text = "@" + shown
    image = _tag_image(text, picture)
    while image.width > TAG_MAX_W and len(shown) > 1:
        shown = shown[:-1].rstrip()
        text = "@" + shown + ELLIPSIS
        image = _tag_image(text, picture)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    _note(dest, text)
    return Overlay(str(dest), TAG_LEFT, TAG_BOTTOM - image.height, 0.0, None)
