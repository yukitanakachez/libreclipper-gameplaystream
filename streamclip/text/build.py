"""All the text for one clip: the Shout captions, the title sticker and the creator tag."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clipper.captions import retime_chunks
from clipper.models import Word
from clipper.pacing import TimeMap
from streamclip.schema import Overlay, Pick, StreamInfo, TextAssets
from streamclip.text import shout, stickers
from streamclip.text.censor import censor


def build_text(clip_dir: Path, pick: Pick, words: list[Word], pcm: np.ndarray, rate: int,
               time_map: TimeMap, info: StreamInfo, avatar: Path | None,
               captions: bool, speed: float) -> TextAssets:
    """Writes captions.ass, title.png and tag.png into `clip_dir` and lists them.

    `words` are in stream seconds; sample 0 of `pcm` is `pick.start`. Times in the result are
    cut time, positions output pixels, paths absolute. No words, or captions off, gives no
    caption file; a title or name with nothing drawable gives no sticker."""
    clip_dir = Path(clip_dir).resolve()
    clip_dir.mkdir(parents=True, exist_ok=True)

    ass: str | None = None
    # Display only: the words handed in, and the transcripts they came from, keep theirs.
    shown = ((w, censor(shout.caption_word(w.text))) for w in words if pick.start <= w.start < pick.end)
    inside = [Word(w.i, text, w.start, w.end, w.prob) for w, text in shown if text]
    if captions and inside:
        scales = shout.word_scales(inside, pcm, rate, pick.start)
        chunks = retime_chunks(shout.chunk(inside), time_map, pick.start)
        if chunks:
            path = clip_dir / "captions.ass"
            path.write_text(shout.build_shout_ass(chunks, scales, time_map.duration))
            ass = str(path)

    overlays: list[Overlay] = []
    title = stickers.title_sticker(pick.title, clip_dir / "title.png", speed)
    if title is not None:
        overlays.append(title)
    tag = stickers.creator_tag(info.creator_name, info.creator_login, avatar, clip_dir / "tag.png")
    if tag is not None:
        overlays.append(tag)
    return TextAssets(ass, overlays)
