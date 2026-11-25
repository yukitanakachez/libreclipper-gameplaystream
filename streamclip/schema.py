"""The data the three stages hand each other. Every class serializes to plain JSON.

Times are in stream seconds unless a field says otherwise. `Rect`, `Word` and `Transcript`
are the podcast clipper's own, reused unchanged; `Rect` values are fractions of the source
frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from clipper.models import Rect, Transcript, Word

__all__ = [
    "Rect", "Word", "Transcript", "Chapter", "StreamInfo", "ViewerClip", "Signals", "Fired",
    "Hotspot", "CandidateIndex", "Beat", "Pick", "CameraBox", "Shot", "Layout", "Overlay",
    "TextAssets", "Sound", "Placed", "MakeOptions", "ClipResult",
]


class _Plain:
    """to_dict and from_dict for a dataclass. Classes holding other classes override from_dict."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


def _rect(d: dict | None) -> Rect | None:
    return None if d is None else Rect.from_dict(d)


@dataclass
class Chapter(_Plain):
    start: float
    end: float
    title: str


@dataclass
class StreamInfo(_Plain):
    source: str
    kind: str                       # "twitch" | "file"
    video_id: str
    title: str
    creator_login: str
    creator_name: str
    duration: float
    started_at: str
    width: int
    height: int
    fps: float
    chapters: list[Chapter]
    extras: dict[str, str]          # "chat" | "viewer_clips" | "avatar" -> "ok" | "missing: <why>"

    @classmethod
    def from_dict(cls, d: dict) -> StreamInfo:
        return cls(**{**d, "chapters": [Chapter.from_dict(c) for c in d["chapters"]],
                      "extras": dict(d["extras"])})


@dataclass
class ViewerClip(_Plain):
    offset: float
    duration: float
    views: int
    title: str


@dataclass
class Signals(_Plain):
    """One value per second for each curve, each 0..1."""
    duration: float
    loudness: list[float]
    surge: list[float]
    voice: list[float]
    chat: list[float] | None        # None when chat was not available
    clips: list[float] | None       # None when viewer clips were not available
    penalty: list[float]
    excitement: list[float]


@dataclass
class Fired(_Plain):
    signal: str                     # loudness | surge | voice | chat | viewer_clip
    t: float
    strength: float
    detail: str


@dataclass
class Hotspot(_Plain):
    n: int
    peak: float
    start: float
    end: float
    score: float
    fired: list[Fired]

    @classmethod
    def from_dict(cls, d: dict) -> Hotspot:
        return cls(**{**d, "fired": [Fired.from_dict(f) for f in d["fired"]]})


@dataclass
class CandidateIndex(_Plain):
    language: str
    captions: bool
    ready: list[int]
    notes: list[str]


@dataclass
class Beat(_Plain):
    kind: str                       # payoff | reaction | hard_cut
    t: float


@dataclass
class Pick(_Plain):
    rank: int
    candidate: int
    start: float
    end: float
    score: int
    reason: str
    title: str
    caption: str
    hashtags: list[str]
    beats: list[Beat]
    protect: list[tuple[float, float]]
    weak: bool

    @classmethod
    def from_dict(cls, d: dict) -> Pick:
        return cls(**{**d, "beats": [Beat.from_dict(b) for b in d["beats"]],
                      "protect": [(float(a), float(b)) for a, b in d["protect"]]})


@dataclass
class CameraBox(_Plain):
    rect: Rect
    confidence: float

    @classmethod
    def from_dict(cls, d: dict) -> CameraBox:
        return cls(Rect.from_dict(d["rect"]), d["confidence"])


@dataclass
class Shot(_Plain):
    start: float                    # source-clip time
    end: float
    kind: str                       # split | face | game
    face: Rect | None
    game: Rect | None
    zoom: Rect | None

    @classmethod
    def from_dict(cls, d: dict) -> Shot:
        return cls(**{**d, "face": _rect(d["face"]), "game": _rect(d["game"]),
                      "zoom": _rect(d["zoom"])})


@dataclass
class Layout(_Plain):
    clip: int
    shots: list[Shot]

    @classmethod
    def from_dict(cls, d: dict) -> Layout:
        return cls(d["clip"], [Shot.from_dict(s) for s in d["shots"]])


@dataclass
class Overlay(_Plain):
    path: str
    x: int                          # output pixels
    y: int
    start: float                    # cut time
    end: float | None               # None: to the end of the clip


@dataclass
class TextAssets(_Plain):
    ass: str | None
    overlays: list[Overlay]

    @classmethod
    def from_dict(cls, d: dict) -> TextAssets:
        return cls(d["ass"], [Overlay.from_dict(o) for o in d["overlays"]])


@dataclass
class Sound(_Plain):
    id: str
    path: str
    role: str                       # a Beat kind
    gain_db: float
    lead: float


@dataclass
class Placed(_Plain):
    sound: str
    path: str
    at: float                       # cut time
    gain_db: float


@dataclass(frozen=True)
class MakeOptions(_Plain):
    speed: float = 1.3
    dead_air: float = 0.5
    sounds: bool = True


@dataclass
class ClipResult(_Plain):
    rank: int
    file: str
    caption_file: str
    title: str
    score: int
    reason: str
    source_start: float
    source_end: float
    duration: float
    weak: bool
    layout: list[str]
    cuts: int
    seconds_cut: float
    sounds: list[str]
    captions: bool
