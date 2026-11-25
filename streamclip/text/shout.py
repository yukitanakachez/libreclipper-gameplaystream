"""The Shout captions: one or two words at a time, tilted Anton capitals, the spoken word in
the accent, and every word sized by how loud it was against the clip's normal speech.

The look comes from the approved mockup ("Shout", option 1): white letters, a thick black
outline and a hard black shadow, never a box behind the words.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from clipper.captions import Chunk, MIN_EVENT, PAUSE, _ass_time, chunk_words, escape_ass
from clipper.models import Word
from streamclip.audio import level_db
from streamclip.settings import ACCENT, CAPTION_Y, OUT_H, OUT_W, SHOUT_MAX, TILT
from streamclip.text import FONT

OUTLINE, SHADOW = 8, 6
X, Y = OUT_W // 2, int(OUT_H * CAPTION_Y)
MAX_WIDTH = 820                 # centred, so it stays clear of the app's buttons on the right
# Room the letters themselves may take: the outline on both sides, the shadow, and what the
# 4 degree tilt adds (about the height of a capital times sin 4 degrees).
TEXT_BUDGET = MAX_WIDTH - 2 * OUTLINE - SHADOW - 20

# How much louder than the clip's median word a word has to be for each size.
STEPS = [(12.0, SHOUT_MAX), (9.0, 140), (6.0, 120)]
SHAKE = r"\t(0,60,\frz8)\t(60,120,\frz1)" + rf"\t(120,180,\frz{TILT})"

# libass sizes a font so that usWinAscent + usWinDescent fill the font size, not the em.
# Anton's are 2876 + 674 on a 2048-unit em, so letters render at 2048 / 3550 of the size.
LIBASS_EM = 2048 / (2876 + 674)
EM = 110                        # letters 110 px to the em, as in the approved mockup
SIZE = round(EM / LIBASS_EM)    # 191: the style's font size in libass's terms


def _bgr(hex_colour: str) -> str:
    r, g, b = hex_colour[1:3], hex_colour[3:5], hex_colour[5:7]
    return f"&H{b}{g}{r}&".upper()


ACCENT_ASS = _bgr(ACCENT)
WHITE_ASS = "&HFFFFFF&"


# A caption word shows its own punctuation (WHAT?, NO!, DON'T) but not the sentence's.
SENTENCE_MARKS = ".,;:\u2026"


def caption_word(text: str) -> str:
    """A word as a caption shows it: full stops, commas, semicolons and colons (and an
    ellipsis) dropped from its end. A word that was only those marks comes back empty."""
    return text.rstrip(SENTENCE_MARKS)


def chunk(words: list[Word]) -> list[Chunk]:
    """One or two words at a time."""
    return chunk_words(words, max_words=2, max_chars=14)


def word_scales(words: list[Word], pcm: np.ndarray, rate: int, clip_start: float) -> dict[int, int]:
    """Each word's size in percent, from its level against the median level of all the words.

    `words` are in stream seconds; sample 0 of `pcm` is `clip_start`. With two words or fewer
    there is no normal to measure against, so nothing is scaled."""
    if len(words) <= 2:
        return {w.i: 100 for w in words}
    levels: dict[int, float | None] = {}
    for w in words:
        a = max(0, int(round((w.start - clip_start) * rate)))
        b = min(len(pcm), int(round((w.end - clip_start) * rate)))
        levels[w.i] = level_db(pcm[a:b]) if b > a else None
    known = [v for v in levels.values() if v is not None]
    if not known:
        return {w.i: 100 for w in words}
    median = float(np.median(known))
    scales = {}
    for i, level in levels.items():
        lift = (level - median) if level is not None else 0.0
        scales[i] = next((pct for step, pct in STEPS if lift >= step), 100)
    return scales


@functools.lru_cache(maxsize=1)
def _font():
    from PIL import ImageFont
    return ImageFont.truetype(str(FONT), SIZE)


def text_width(text: str, percent: int = 100) -> float:
    """Width of `text` as libass draws it in the Shout style at `percent`, in frame pixels."""
    return _font().getlength(text) * LIBASS_EM * percent / 100


def _line_width(words: list[str], scales: list[int]) -> float:
    # A word's following space is drawn at that word's scale.
    return sum(text_width(w + (" " if k + 1 < len(words) else ""), s)
               for k, (w, s) in enumerate(zip(words, scales)))


def _fit(words: list[str], scales: list[int]) -> tuple[list[int], int]:
    """Scales and font size so the line fits: the scales come down together first, never under
    100, then the font size."""
    if _line_width(words, scales) <= TEXT_BUDGET:
        return scales, SIZE
    lo, hi = 0.0, 1.0                       # the largest factor on the extra size that fits
    for _ in range(20):
        mid = (lo + hi) / 2
        trial = [100 + int((s - 100) * mid) for s in scales]
        lo, hi = (mid, hi) if _line_width(words, trial) <= TEXT_BUDGET else (lo, mid)
    fitted = [100 + int((s - 100) * lo) for s in scales]
    width = _line_width(words, fitted)
    if width <= TEXT_BUDGET:
        return fitted, SIZE
    return fitted, max(1, math.floor(SIZE * TEXT_BUDGET / width))


HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Shout,Anton,{SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{OUTLINE},{SHADOW},5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_shout_ass(chunks: list[Chunk], scales: dict[int, int], duration: float) -> str:
    """The captions file. `chunks` are already in cut time; nothing runs past `duration`."""
    lines: list[str] = []
    for n, ch in enumerate(chunks):
        shown = [escape_ass(w.text.upper()) for w in ch.words]
        sizes, font_size = _fit(shown, [scales.get(w.i, 100) for w in ch.words])
        loud = any(scales.get(w.i, 100) >= SHOUT_MAX for w in ch.words)
        chunk_end = ch.end
        if n + 1 < len(chunks) and chunks[n + 1].start - chunk_end <= PAUSE:
            chunk_end = chunks[n + 1].start          # no flicker between chunks
        for k, word in enumerate(ch.words):
            start = max(ch.start, word.start)
            hand_over = ch.words[k + 1].start if k + 1 < len(ch.words) else chunk_end
            end = min(max(start + MIN_EVENT, hand_over), duration)
            if start >= end:
                continue
            head = rf"\an5\pos({X},{Y})\frz{TILT}"
            if font_size != SIZE:
                head += rf"\fs{font_size}"
            if loud and scales.get(word.i, 100) >= SHOUT_MAX:
                head += SHAKE
            body = ""
            for j, (text, size) in enumerate(zip(shown, sizes)):
                colour = ACCENT_ASS if loud or j == k else WHITE_ASS
                space = " " if j + 1 < len(shown) else ""
                body += rf"{{\c{colour}\fscx{size}\fscy{size}}}{text}{space}"
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Shout,,0,0,0,,"
                         f"{{{head}}}{body}")
    return HEADER + "".join(line + "\n" for line in lines)
