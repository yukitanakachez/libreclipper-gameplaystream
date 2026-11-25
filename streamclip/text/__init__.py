"""Everything written on a clip: the Shout captions, the title sticker and the creator tag.

Captions are subtitles (ASS) burned in by libass, timed in cut time. The title sticker and the
creator tag are PNG images made with Pillow and laid over the video by the render lane.
"""

from __future__ import annotations

from pathlib import Path

FONTS_DIR = Path(__file__).parent.parent / "assets" / "fonts"
FONT = FONTS_DIR / "Anton-Regular.ttf"

from streamclip.text.build import build_text  # noqa: E402  (needs FONT above)

__all__ = ["FONTS_DIR", "FONT", "build_text"]
