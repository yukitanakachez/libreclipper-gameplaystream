"""The sound pack: one short synthesised sound for each kind of beat the judge marks."""

from __future__ import annotations

import json
from pathlib import Path

from streamclip.schema import Sound

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "sounds"

__all__ = ["load_pack"]


def load_pack() -> dict[str, Sound]:
    """The pack keyed by role (a Beat kind), with absolute paths to the files."""
    rows = json.loads((ASSETS / "manifest.json").read_text())
    return {row["role"]: Sound(id=row["id"], path=str(ASSETS / row["file"]), role=row["role"],
                               gain_db=float(row["gain_db"]), lead=float(row["lead"]))
            for row in rows}
