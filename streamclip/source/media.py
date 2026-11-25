"""ffmpeg steps both sources share: probing, cutting an exact stretch, taking the sound."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from clipper import shell
from streamclip.errors import NoVideo
from streamclip.settings import processors

Runner = Callable[..., str]

AUDIO_BITRATE = "96k"
NO_VIDEO = "The video for that stretch came back as sound only."


def probe(path: Path, run: Runner = shell.run) -> dict:
    """ffprobe's streams and format for `path`."""
    return json.loads(run([shell.ffprobe(), "-v", "error", "-print_format", "json",
                           "-show_streams", "-show_format", str(path)]))


def streams_of(info: dict, kind: str) -> list[dict]:
    return [s for s in info.get("streams", []) if s.get("codec_type") == kind]


def frame_rate(stream: dict) -> float:
    """Frames per second from ffprobe's "30000/1001" style fields; 0.0 when unknown."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = str(stream.get(key, "0/0")).partition("/")
        try:
            value = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if value > 0:
            return value
    return 0.0


@contextmanager
def partial(dest: Path) -> Iterator[Path]:
    """A temporary name beside `dest`, moved into place only when the block succeeds, so a
    stage that is stopped half-way never leaves a short file that looks finished."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    try:
        yield tmp
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def scale_filter(max_height: int) -> str:
    """Down to `max_height` at most, never up, keeping the aspect with even sizes."""
    h = int(max_height)
    return f"scale=-2:'trunc(min(ih\\,{h})/2)*2'"


def check_stretch(start: float, end: float) -> None:
    """A stretch starts at 0 or later and has length. Anything else is a caller's mistake:
    clamping it here would move t=0 away from `start` without a sign."""
    if not (start >= 0.0 and end > start):
        raise ValueError(f"a stretch needs 0 <= start < end, got {start!r} to {end!r}")


def cut_stretch(src: Path, offset: float, length: float, max_height: int, dest: Path,
                run: Runner = shell.run) -> Path:
    """Re-encode `length` seconds of `src` from `offset`, scaled, with its sound.

    `-ss` before `-i` with a re-encode is frame-exact in current ffmpeg, so t=0 of the
    result is `offset` of `src`. Raises NoVideo when the result has no picture."""
    with partial(dest) as tmp:
        run([shell.ffmpeg(), "-v", "error", "-nostdin", "-y",
             "-ss", f"{max(0.0, offset):.3f}", "-i", str(src), "-t", f"{length:.3f}",
             "-map", "0:v:0?", "-map", "0:a:0?", "-sn", "-dn",
             "-vf", scale_filter(max_height),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-threads", str(processors()), "-c:a", "aac", "-b:a", "160k",
             "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(tmp)])
        if not streams_of(probe(tmp, run), "video"):
            raise NoVideo(NO_VIDEO)
    return Path(dest)


def take_audio(src: Path, dest: Path, run: Runner = shell.run) -> Path:
    """The whole sound of `src` as mono AAC."""
    with partial(dest) as tmp:
        run([shell.ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", str(src),
             "-vn", "-sn", "-dn", "-map", "0:a:0", "-ac", "1", "-c:a", "aac",
             "-b:a", AUDIO_BITRATE, "-movflags", "+faststart", str(tmp)])
    return Path(dest)
