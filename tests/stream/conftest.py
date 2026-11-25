"""Fixtures every stream lane shares.

`synthetic_stream` is the stand-in broadcast described in the master plan under "The test
fixture": two minutes with two loud moments (30-32 s, 95-97 s) and one stretch of dead air
(70-80 s). Change it only together with that section, because other lanes' tests lean on it.
"""

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipper import shell

DURATION = 120
SIZE = "1280x720"
FPS = 30

# Sound, as RMS levels of the mono mix `audio.read_pcm` returns: noise at -40 dB throughout, a 440 Hz tone at -20 dB, a 330 Hz square
# wave at -3 dB for the loud bursts. Uniform noise of peak a has an RMS of a/sqrt(3), a sine
# of peak a has a/sqrt(2), a square wave has a.
_NOISE = "0.01732*(2*random(0)-1)"
_TONE = "0.14142*sin(2*PI*440*t)*(between(t,10,14)+between(t,20,26)+between(t,50,58))"
_BURST = "0.70795*(2*gt(sin(2*PI*330*t),0)-1)*(between(t,30,32)+between(t,95,97))"

_GRAPH = (
    # A test pattern scrolling sideways, 6.4 pixels a frame, so the picture changes strongly.
    f"testsrc2=s={SIZE}:r={FPS},scroll=h=0.005[moving];"
    # 70-80 s: one still frame laid over everything, so the whole picture is frozen.
    f"testsrc2=s={SIZE}:r={FPS},trim=end_frame=1,loop=loop=-1:size=1[still];"
    "[moving][still]overlay=enable='between(t,70,80)'[base];"
    # The stand-in camera: colour bars, 320x180 at 24,24.
    f"smptebars=s=320x180:r={FPS}[camera];"
    "[base][camera]overlay=x=24:y=24[v];"
    # The noise is made at 16 kHz so it measures the same at any rate a test reads it at.
    f"aevalsrc='{_NOISE}':s=16000,aresample=48000[noise];"
    f"aevalsrc='{_TONE}+{_BURST}':s=48000[marks];"
    "[noise][marks]amix=inputs=2:normalize=0,"
    "volume=0:enable='between(t,70,80)',"
    # ffmpeg folds stereo to mono at 0.707 a side, which lifts identical sides by 3 dB. Each
    # side is written 3 dB down, so the levels above are what `audio.read_pcm` measures.
    "pan=stereo|c0=0.70711*c0|c1=0.70711*c0[a]"
)


def _stream_command(dest: Path) -> list[str]:
    return [shell.ffmpeg(), "-v", "error", "-y", "-filter_complex", _GRAPH,
            "-map", "[v]", "-map", "[a]", "-t", str(DURATION),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-g", str(FPS),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-movflags", "+faststart", str(dest)]


@pytest.fixture(scope="session")
def synthetic_stream(request, tmp_path_factory) -> Path:
    """The two-minute stand-in stream. Built once and kept in pytest's cache folder."""
    recipe = hashlib.sha256(repr(_stream_command(Path("x"))[1:]).encode()).hexdigest()[:12]
    cache = getattr(request.config, "cache", None)       # absent under -p no:cacheprovider
    folder = cache.mkdir("streamclip") if cache else tmp_path_factory.mktemp("streamclip")
    path = Path(folder) / f"synthetic-{recipe}.mp4"
    if path.exists():
        return path
    partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.mp4")
    try:
        subprocess.run(_stream_command(partial), check=True, capture_output=True, timeout=600)
    except OSError as exc:                      # no ffmpeg on this machine
        pytest.skip(f"ffmpeg is needed to build the test stream: {exc}")
    except subprocess.CalledProcessError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg could not build the test stream: "
                           + exc.stderr.decode(errors="replace")[-500:]) from exc
    os.replace(partial, path)
    return path


@pytest.fixture(scope="session")
def mean_frame_difference():
    """mean_frame_difference(video, start, end): how much the picture changes, 0 to 255.

    Frames ten times a second, shrunk to 160x90 grey; the mean absolute difference between
    neighbours."""
    def measure(video: Path, start: float, end: float) -> float:
        import cv2

        from clipper.reframe.faces import iter_sampled_frames
        small = [cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_RGB2GRAY).astype(np.float32)
                 for _, frame in iter_sampled_frames(Path(video), start, end, fps=10.0)]
        return float(np.mean([np.abs(b - a).mean() for a, b in zip(small, small[1:])]))
    return measure
