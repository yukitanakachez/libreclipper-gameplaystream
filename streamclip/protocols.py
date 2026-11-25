"""What a stream source can do, whether it is a Twitch broadcast or a file on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from streamclip.schema import StreamInfo, ViewerClip


@runtime_checkable
class Source(Protocol):
    def describe(self) -> StreamInfo:
        """Who and what the stream is. Raises Refused when it cannot be clipped."""
        ...

    def fetch_audio(self, dest: Path) -> Path:
        """The whole stream's sound, written to `dest`."""
        ...

    def fetch_video(self, start: float, end: float, max_height: int, dest: Path) -> Path:
        """One stretch of video with its sound. t=0 is `start` within 0.05 s.
        Raises NoVideo when what came back has no video track."""
        ...

    # The extras never fail a job: each gives (value or None, "ok" | "missing: <why>").

    def chat_times(self) -> tuple[list[float] | None, str]:
        ...

    def viewer_clips(self) -> tuple[list[ViewerClip] | None, str]:
        ...

    def avatar(self, dest: Path) -> tuple[Path | None, str]:
        ...
