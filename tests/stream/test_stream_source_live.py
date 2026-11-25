"""Against a real public broadcast. Deselected by default; run by hand with

    STREAMCLIP_LIVE_VOD=https://www.twitch.tv/videos/<id> pytest tests/stream/test_stream_source_live.py -m live -q

Downloads stay small: a 20-second video stretch and about 30 seconds of sound, deleted with
the test's temporary folder. With no STREAMCLIP_LIVE_VOD, the newest past broadcast of a
few busy channels is used."""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from clipper import shell
from streamclip import audio
from streamclip.source import open_source
from streamclip.source.twitch import yt_dlp

pytestmark = pytest.mark.live

CHANNELS = ["jynxzi", "xqc", "kaicenat", "caseoh_"]
START, END = 600.0, 620.0
RATE = 8000


def _link() -> str:
    if os.environ.get("STREAMCLIP_LIVE_VOD"):
        return os.environ["STREAMCLIP_LIVE_VOD"]
    for channel in CHANNELS:
        out = subprocess.run(
            yt_dlp("--flat-playlist", "--playlist-end", "1", "--print", "id",
                   f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time"),
            capture_output=True, text=True, timeout=120).stdout.split()
        if out:
            return f"https://www.twitch.tv/videos/{out[0].lstrip('v')}"
    pytest.skip("no public past broadcast found")


def _reference(link: str, dest: Path) -> tuple[Path, float]:
    """The sound-only playlist's own segments around the stretch, and the stream time of
    their first sample: each segment's place comes from the #EXTINF lengths before it, the
    clock Twitch's own player uses, with no cutting involved."""
    playlist = subprocess.run(yt_dlp("-g", "-f", "Audio_Only", link), capture_output=True,
                              text=True, timeout=120, check=True).stdout.strip()
    text = urllib.request.urlopen(playlist, timeout=60).read().decode()
    base = playlist.rsplit("/", 1)[0]
    t, wanted, first = 0.0, [], None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#EXTINF:"):
            length = float(line.split(":")[1].split(",")[0])
            name = lines[i + 1].strip()
            if t + length > START - 5 and t < END + 5:
                wanted.append(name)
                first = t if first is None else first
            t += length
    with dest.open("wb") as fh:
        for name in wanted:
            fh.write(urllib.request.urlopen(f"{base}/{name}", timeout=60).read())
    return dest, first


def test_fetched_stretch_lines_up_with_the_sound(tmp_path):
    link = _link()
    src = open_source(link)
    info = src.describe()
    assert info.duration > END + 10
    stretch = src.fetch_video(START, END, 360, tmp_path / "w.mp4")
    video = next(s for s in shell.probe(stretch)["streams"] if s["codec_type"] == "video")
    assert int(video["height"]) <= 360
    assert abs(float(shell.probe(stretch)["format"]["duration"]) - (END - START)) < 0.15

    ref_path, ref_start = _reference(src.url, tmp_path / "ref.ts")
    ref = audio.read_pcm(ref_path, rate=RATE)
    probe = shell.probe(ref_path)
    lead = float(probe["streams"][0].get("start_time") or 0) - float(probe["format"]["start_time"])
    ref_start += lead
    x = audio.read_pcm(stretch, 1.0, 15.0, rate=RATE)
    corr = np.correlate(ref, x, mode="valid")
    real = ref_start + int(np.argmax(corr)) / RATE - 1.0
    lag = real - START
    print(f"\n{link}: stretch starts at {real:.3f} s, asked {START}, lag {lag:+.3f} s")
    assert abs(lag) < 0.05


def test_viewer_clips_and_picture(tmp_path):
    src = open_source(_link())
    info = src.describe()
    clips, note = src.viewer_clips()
    print(f"\nviewer clips: {note}, {len(clips or [])} found")
    assert note == "ok" and all(0 <= c.offset <= info.duration for c in clips)
    path, note = src.avatar(tmp_path / "avatar.jpg")
    assert note == "ok" and path.stat().st_size > 1000
