"""One folder per job, plain JSON files as the only state, as the podcast clipper does.

    <job>/
      request.json  status.json  stream.json
      source/       audio.m4a, chat.json, viewer_clips.json, avatar.jpg
      signals.json  hotspots.json
      candidates/   index.json, NN/{transcript.json, window.mp4, sheet.jpg}
      judge_raw.json  picks.json  camera.json
      clips/NN/     source.mp4, layout.json, cut.json, captions.ass, title.png, tag.png
      out/          NN-slug.mp4, NN-slug.txt, clips.json

NN is the hot spot's number under candidates/ and the pick's rank under clips/ and out/.
Nothing here creates a folder; `clipper.jobs.write_json` makes the parents it needs.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from clipper.jobs import read_json, write_json

STAGES = ("find", "judge", "make", "done", "failed")
NEW = "new"                         # get_status() before any stage has written a status


def nn(n: int) -> str:
    return f"{n:02d}"


def slug(title: str) -> str:
    """A safe file name from any title: lowercase ASCII letters, digits and dashes."""
    plain = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")[:50].rstrip("-") or "clip"


class JobDir:
    def __init__(self, root: Path, out: Path | None = None) -> None:
        self.root = Path(root)
        self._out = Path(out) if out is not None else None

    def __repr__(self) -> str:
        return f"JobDir({str(self.root)!r})"

    @property
    def request(self) -> Path:
        return self.root / "request.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def stream(self) -> Path:
        return self.root / "stream.json"

    @property
    def audio(self) -> Path:
        return self.root / "source" / "audio.m4a"

    @property
    def chat(self) -> Path:
        return self.root / "source" / "chat.json"

    @property
    def viewer_clips(self) -> Path:
        return self.root / "source" / "viewer_clips.json"

    @property
    def avatar(self) -> Path:
        return self.root / "source" / "avatar.jpg"

    @property
    def signals(self) -> Path:
        return self.root / "signals.json"

    @property
    def hotspots(self) -> Path:
        return self.root / "hotspots.json"

    @property
    def candidates_index(self) -> Path:
        return self.root / "candidates" / "index.json"

    @property
    def judge_raw(self) -> Path:
        return self.root / "judge_raw.json"

    @property
    def picks(self) -> Path:
        return self.root / "picks.json"

    @property
    def camera(self) -> Path:
        return self.root / "camera.json"

    @property
    def out(self) -> Path:
        return self._out if self._out is not None else self.root / "out"

    def candidate(self, n: int) -> Path:
        return self.root / "candidates" / nn(n)

    def clip(self, rank: int) -> Path:
        return self.root / "clips" / nn(rank)

    def set_status(self, stage: str, step: str, detail: str = "", error: str | None = None) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        write_json(self.status, {"stage": stage, "step": step, "detail": detail, "error": error,
                                 "updated": datetime.now(UTC).isoformat()})

    def get_status(self) -> dict:
        if not self.status.exists():
            return {"stage": NEW, "step": "", "detail": "", "error": None, "updated": ""}
        return read_json(self.status)
