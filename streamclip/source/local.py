"""A video file on disk. It has no creator, no chat, no viewer clips and no picture."""

from __future__ import annotations

from pathlib import Path

from clipper import shell
from streamclip.errors import Refused
from streamclip.schema import StreamInfo, ViewerClip
from streamclip.settings import MAX_STREAM
from streamclip.source import media

NOT_TWITCH = "missing: not a Twitch broadcast"
EXTRAS = ("chat", "viewer_clips", "avatar")


class FileSource:
    def __init__(self, path: str | Path, run=shell.run) -> None:
        self.path = Path(path).expanduser()
        self.run = run

    def describe(self) -> StreamInfo:
        if not self.path.is_file():
            raise Refused(f"No file at {self.path}.")
        try:
            info = media.probe(self.path, self.run)
        except (shell.CommandError, ValueError) as exc:
            raise Refused("That file could not be read as a video.") from exc
        videos = [s for s in media.streams_of(info, "video")
                  if not (s.get("disposition") or {}).get("attached_pic")]
        if not videos:
            raise Refused("That file has no video, only sound. Send a video file.")
        if not media.streams_of(info, "audio"):
            raise Refused("That file has no sound. Clips are found by their sound, so the "
                          "recording needs it.")
        try:
            duration = float(info.get("format", {}).get("duration") or 0.0)
        except ValueError:
            duration = 0.0
        if duration <= 0:
            raise Refused("That file could not be read as a video.")
        if duration > MAX_STREAM:
            raise Refused("That recording is over 6 hours long. The limit is 6 hours.")
        video = videos[0]
        return StreamInfo(
            source=str(self.path), kind="file", video_id="", title=self.path.stem,
            creator_login="", creator_name="", duration=duration, started_at="",
            width=int(video.get("width") or 0), height=int(video.get("height") or 0),
            fps=media.frame_rate(video), chapters=[],
            extras={name: NOT_TWITCH for name in EXTRAS})

    def fetch_audio(self, dest: Path) -> Path:
        return media.take_audio(self.path, dest, self.run)

    def fetch_video(self, start: float, end: float, max_height: int, dest: Path) -> Path:
        media.check_stretch(start, end)
        return media.cut_stretch(self.path, start, end - start, max_height, dest, self.run)

    def chat_times(self) -> tuple[list[float] | None, str]:
        return None, NOT_TWITCH

    def viewer_clips(self) -> tuple[list[ViewerClip] | None, str]:
        return None, NOT_TWITCH

    def avatar(self, dest: Path) -> tuple[Path | None, str]:
        return None, NOT_TWITCH
