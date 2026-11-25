"""The fixed values every lane shares. Tuned numbers live with the code they tune."""

from __future__ import annotations

import os


def processors() -> int:
    """How many processors this job may use, which can be fewer than the machine reports:
    a Starwish job machine shows 8 and allows 4. Every thread count comes from here."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):       # no such call on macOS or Windows
        return max(1, os.cpu_count() or 1)


def _setting(name: str, default: str) -> str:
    """A value that STREAMCLIP_<name> in the environment may replace."""
    return os.environ.get(f"STREAMCLIP_{name}", "").strip() or default


def _count(name: str, default: int) -> int:
    try:
        return max(1, int(_setting(name, str(default))))
    except ValueError:
        return default


# Speech to text, in two passes. Every hot spot's window is heard once, fast, so the judge
# has something to read: Starwish's slower machine needs 23 minutes for 36 minutes of sound
# with `base` at beam 5, and 47 with `small`. Only the stretches that become clips are heard
# again, accurately, and those words make the captions.
WINDOW_MODEL = _setting("WINDOW_MODEL", _setting("WHISPER", "base"))    # STREAMCLIP_WHISPER: its older name
WINDOW_BEAM = _count("WINDOW_BEAM", 1)
CAPTION_MODEL = _setting("CAPTION_MODEL", "small")
CAPTION_BEAM = _count("CAPTION_BEAM", 5)
CAPTION_PAD = 1.0               # seconds heard either side of a clip on the second pass

# Output frame: face panel on top, game panel below.
OUT_W, OUT_H = 1080, 1920
# Both heights even: the picture is yuv420p, which stores colour for pairs of rows, and
# stacking an odd-height panel (the spec's 691) overruns ffmpeg's vstack buffer. glibc aborts
# on that ("malloc(): corrupted top size", Starwish dev runs 2 and 3); macOS lets it pass.
FACE_H, GAME_H = 692, 1228

# Pacing.
SPEED = 1.3                     # pitch kept
DEAD_AIR = 0.5                  # seconds of dead air before it is cut
BREATH = 0.1                    # what a cut stretch is shortened to
MIN_FINISHED, MAX_FINISHED = 15.0, 40.0     # finished seconds

# How many clips.
DEFAULT_CLIPS, MIN_CLIPS, MAX_CLIPS = 6, 1, 10
FLOOR = 5                       # never fewer on a normal stream; the weaker ones are marked
SHORT_STREAM = 45 * 60          # under this the floor is one clip per SHORT_STREAM_PER_CLIP
SHORT_STREAM_PER_CLIP = 8 * 60
MAX_STREAM = 6 * 3600           # longer streams are refused

# Hot spots.
PEAK_GAP = 45.0                 # seconds between peaks, at least
PEAKS_PER_CLIP = 4
WINDOW_BEFORE, WINDOW_AFTER = 60.0, 30.0

# The look.
ACCENT = "#D7FF3E"
TILT = 4                        # degrees
SHOUT_MAX = 165                 # percent of the normal caption size
TITLE_SECONDS = 3.0             # finished seconds the title sticker stays up
SAFE_TOP, SAFE_BOTTOM, SAFE_RIGHT = 0.10, 0.20, 0.12   # under the app's own buttons
CAPTION_Y = 0.71                # caption line, as a fraction of the height

# Sounds.
MAX_SOUNDS = 5                  # per clip
SOUND_GAP = 4.0                 # finished seconds between sounds, at least
LOUDNESS = "I=-14:TP=-1:LRA=11"
