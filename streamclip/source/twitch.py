"""A Twitch past broadcast, fetched the way yt-dlp fetches it: no key, no login, no cookies.

yt-dlp runs as `python -m yt_dlp` through an injectable runner, so tests pass a fake and
never touch the network. A link's shape is checked before anything is fetched."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from clipper import shell
from streamclip.errors import NoVideo, Refused, StageError
from streamclip.schema import Chapter, StreamInfo, ViewerClip
from streamclip.settings import MAX_STREAM
from streamclip.source import extras, media

DESCRIBE_TIMEOUT = 180
AUDIO_TIMEOUT = 3 * 3600        # the longest broadcast's sound, with room to spare
VIDEO_TIMEOUT = 900             # one stretch
FRAGMENTS = 8                   # HLS pieces fetched at once
PAD = 5.0                       # seconds fetched beyond each side of a stretch
NETWORK_TRIES = 3               # runs of the downloader when Twitch stalls or drops the line
RETRY_WAIT = 5.0                # seconds before the first rerun; doubled each time

CHANNEL = ("That link is a channel, not a past broadcast. Send the link of one finished "
           "broadcast, like twitch.tv/videos/123456.")
NOT_BROADCAST = ("That link is not a past broadcast. Send the link of one finished "
                 "broadcast, like twitch.tv/videos/123456.")
CLIP = "That link is a single clip. Send the link of the whole past broadcast."
SUBSCRIBERS = "That broadcast is for subscribers only."
GONE = "That broadcast no longer exists. Twitch deletes past broadcasts after 7 to 60 days."
LIVE = "That broadcast is still live. It can be clipped once it has ended."
TOO_LONG = "That broadcast is over 6 hours long. The limit is 6 hours."
NOT_FETCHED = "missing: not fetched yet"
NO_VIDEO = "That broadcast has no video that can be fetched without a login."

# What the downloader says, matched loosely, and what the customer is told instead.
_KNOWN_FAILURES = [
    (re.compile(r"subscriber|vod_manifest_restricted|unauthorized_entitlements", re.I), SUBSCRIBERS),
    (re.compile(r"does not exist|HTTP Error 404", re.I), GONE),
]

# A stall or a dropped connection, which yt-dlp's own retries do not cover while it fetches
# the broadcast's playlist (usher.ttvnw.net). These pass, so the run is tried again.
_NETWORK_HICCUP = re.compile(
    r"timed out|TransportError|Connection (?:reset|aborted|refused)|RemoteDisconnected|"
    r"IncompleteRead|Temporary failure in name resolution|EOF occurred", re.I)

# Path segments that are Twitch pages, not channel names.
_NOT_CHANNELS = {"directory", "downloads", "jobs", "p", "settings", "subscriptions", "search",
                 "turbo", "wallet", "inventory", "drops", "friends", "messages", "store"}


def broadcast_id(link: str) -> str:
    """The broadcast's number from a twitch.tv link, or Refused with why the link is wrong.

    Accepts twitch.tv/videos/<digits> and twitch.tv/<channel>/v/<digits> on any twitch.tv
    host. Nothing is fetched."""
    parsed = urlparse(link if "://" in link else f"https://{link}")
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host.startswith("clips."):
        raise Refused(CLIP)
    if len(parts) >= 2 and parts[0] == "videos" and parts[1].isdigit():
        return parts[1]
    if len(parts) >= 3 and parts[1] == "v" and parts[2].isdigit():
        return parts[2]
    if len(parts) >= 2 and parts[1] in ("clip", "clips") and len(parts) >= 3:
        raise Refused(CLIP)
    if parts and parts[0].lower() not in _NOT_CHANNELS and parts[0] != "videos" and (
            len(parts) == 1 or parts[1] in ("videos", "about", "schedule", "clips", "home")):
        raise Refused(CHANNEL)
    raise Refused(NOT_BROADCAST)


def yt_dlp(*args: str) -> list[str]:
    """The downloader's command line, run as a module of this Python."""
    return [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-progress",
            "--extractor-retries", "3", "--retries", "5", "--fragment-retries", "10",
            "--socket-timeout", "30", *args]


def plain_failure(exc: Exception) -> Exception:
    """A downloader failure as Refused (a known, customer-readable reason) or StageError."""
    if isinstance(exc, subprocess.TimeoutExpired):
        return StageError("Twitch took too long to answer.")
    if isinstance(exc, shell.CommandError):
        for pattern, reason in _KNOWN_FAILURES:
            if pattern.search(exc.stderr or ""):
                return Refused(reason)
        lines = [l.strip() for l in (exc.stderr or "").splitlines() if l.strip()]
        errors = [l for l in lines if l.startswith("ERROR:")]
        if errors:
            return StageError(f"yt_dlp failed: {errors[-1][:300]}")
    return StageError(shell.describe_failure(exc))


def _is_video(fmt: dict) -> bool:
    return fmt.get("vcodec") not in (None, "none") and bool(fmt.get("height"))


def best_format(formats: list[dict], max_height: int = 1080) -> dict | None:
    """The tallest, then smoothest, video format at `max_height` or under."""
    fitting = [f for f in formats if _is_video(f) and int(f["height"]) <= max_height]
    if not fitting:
        return None
    return max(fitting, key=lambda f: (int(f["height"]), float(f.get("fps") or 0)))


def _even(x: float) -> int:
    return int(round(x / 2.0)) * 2


class TwitchSource:
    def __init__(self, link: str, run=shell.run, fetch=None) -> None:
        self.link = link if "://" in link else f"https://{link}"
        self.run = run
        self.fetch = fetch          # None: plain HTTP, looked up when an extra runs
        self._info: StreamInfo | None = None
        self._notes: dict[str, str] = {}    # what each extra came to, once it has run
        try:
            self.video_id: str | None = broadcast_id(self.link)
        except Refused:
            self.video_id = None        # describe() says why

    @property
    def url(self) -> str:
        """The broadcast's canonical link, which is what the downloader is given."""
        return f"https://www.twitch.tv/videos/{self.video_id}"

    def _ytdlp(self, *args: str, timeout: int) -> str:
        """Run the downloader, again after a network hiccup, up to NETWORK_TRIES times."""
        for attempt in range(NETWORK_TRIES):
            try:
                return self.run(yt_dlp(*args), timeout=timeout)
            except shell.CommandError as exc:
                last = attempt == NETWORK_TRIES - 1
                if last or not _NETWORK_HICCUP.search(exc.stderr or ""):
                    raise plain_failure(exc) from exc
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise plain_failure(exc) from exc
            time.sleep(RETRY_WAIT * 2 ** attempt)
        raise AssertionError("unreachable")

    def describe(self) -> StreamInfo:
        if self._info is not None:
            return self._info
        vid = broadcast_id(self.link)       # raises Refused before any fetch
        answer = self._ytdlp("-J", "--no-playlist", self.url, timeout=DESCRIBE_TIMEOUT)
        try:
            data = json.loads(answer)
        except (TypeError, ValueError) as exc:
            raise StageError("Twitch's answer about that broadcast could not be read.") from exc
        if data.get("live_status") in ("is_live", "is_upcoming") or data.get("is_live"):
            raise Refused(LIVE)
        duration = float(data.get("duration") or 0.0)
        if duration <= 0:
            raise StageError("Twitch did not say how long that broadcast is.")
        if duration > MAX_STREAM:
            raise Refused(TOO_LONG)
        best = best_format(data.get("formats") or [])
        if best is None:
            raise Refused(NO_VIDEO)
        height = int(best["height"])
        width = int(best.get("width") or _even(height * 16 / 9))
        started = data.get("timestamp")
        started_at = (datetime.fromtimestamp(float(started), tz=timezone.utc).isoformat()
                      if started else "")
        chapters = [Chapter(start=float(c.get("start_time") or 0.0),
                            end=float(c.get("end_time") or duration),
                            title=str(c.get("title") or ""))
                    for c in data.get("chapters") or []]
        login = str(data.get("uploader_id") or "")
        self._info = StreamInfo(
            source=self.link, kind="twitch", video_id=vid, title=str(data.get("title") or ""),
            creator_login=login, creator_name=str(data.get("uploader") or login),
            duration=duration, started_at=started_at, width=width, height=height,
            fps=float(best.get("fps") or 0.0), chapters=chapters,
            extras={key: self._notes.get(key, NOT_FETCHED)
                    for key in ("chat", "viewer_clips", "avatar")})
        return self._info

    def _download(self, fmt: str, piece: Path, *args: str, timeout: int) -> Path:
        """Run the downloader into `piece`; Refused or StageError on failure."""
        self._ytdlp("-f", fmt, "--no-playlist", "--force-overwrites", "--no-cache-dir",
                    "--concurrent-fragments", str(FRAGMENTS),
                    "--ffmpeg-location", shell.ffmpeg(), *args, "-o", str(piece), self.url,
                    timeout=timeout)
        if not piece.is_file():
            raise StageError("The download finished without writing a file.")
        return piece

    def fetch_audio(self, dest: Path) -> Path:
        """The whole broadcast's sound, as mono AAC like a file source gives."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".fetch-") as work:
            piece = self._download("Audio_Only/bestaudio", Path(work) / "sound.m4a",
                                   timeout=AUDIO_TIMEOUT)
            return media.take_audio(piece, dest, self.run)

    def fetch_video(self, start: float, end: float, max_height: int, dest: Path) -> Path:
        """One stretch at `max_height` or under, t=0 exactly at `start`.

        Twitch cuts land on keyframes, so PAD seconds more are fetched each side. yt-dlp
        cuts with ffmpeg's -ss and a stream copy, which keeps the frames before the cut in
        the file but starts its clock exactly at the section's start (measured on a live
        broadcast: within one frame), so the piece's t=0 is `first` and the exact stretch
        is cut from `start - first` with a re-encode."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        media.check_stretch(start, end)
        # The section is sent to a tenth of a second, so `first` is exactly what is sent and
        # the trim below starts from the piece's true t=0.
        first = max(0.0, math.floor((start - PAD) * 10) / 10)
        last = math.ceil((end + PAD) * 10) / 10
        if self._info is not None:
            last = min(last, self._info.duration)
        h = int(max_height)
        fmt = f"best[height<={h}][vcodec!=none]/worst[vcodec!=none]/best"
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".fetch-") as work:
            piece = self._download(fmt, Path(work) / "piece.mp4",
                                   "--download-sections", f"*{first:.1f}-{last:.1f}",
                                   timeout=VIDEO_TIMEOUT)
            try:
                return media.cut_stretch(piece, start - first, end - start, h, dest, self.run)
            except shell.CommandError as exc:
                if not media.streams_of(media.probe(piece, self.run), "video"):
                    raise NoVideo(media.NO_VIDEO) from exc
                raise StageError(shell.describe_failure(exc)) from exc

    def _extra(self, key: str, name: str, fn) -> tuple[object | None, str]:
        """Run one extra through extras.optional and note the outcome in the description."""
        value, note = extras.optional(fn, name)
        self._notes[key] = note
        if self._info is not None:
            self._info.extras[key] = note
        return value, note

    def _web(self):
        return self.fetch or extras.http_get

    def chat_times(self) -> tuple[list[float] | None, str]:
        def fn():
            tool = extras.chat_tool()
            if not tool:
                raise extras.Missing("chat tool not installed")
            return extras.download_chat(broadcast_id(self.link), self.run, tool)
        return self._extra("chat", "chat", fn)

    def viewer_clips(self) -> tuple[list[ViewerClip] | None, str]:
        def fn():
            info = self.describe()
            return extras.fetch_viewer_clips(info.creator_login, info.video_id,
                                             extras.started_epoch(info.started_at), self._web())
        return self._extra("viewer_clips", "viewer clips", fn)

    def avatar(self, dest: Path) -> tuple[Path | None, str]:
        def fn():
            return extras.fetch_avatar(self.describe().creator_login, Path(dest), self._web())
        return self._extra("avatar", "picture", fn)
