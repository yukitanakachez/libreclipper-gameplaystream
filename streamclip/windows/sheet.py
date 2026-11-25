"""One contact sheet per window: 15 frames on a 5×3 grid, each marked with its time, so the
judge can see what happened on screen while it reads the transcript."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from clipper import shell
from clipper.reframe.faces import iter_sampled_frames

COLS, ROWS = 5, 3
CELL_W, CELL_H = 384, 216
TAB_H = 22                      # the black time tab at each cell's top left
QUALITY = 80


def frame_times(duration: float, frames: int) -> list[float]:
    """The middle of each of `frames` equal slices of the video, in seconds."""
    if duration <= 0 or frames <= 0:
        return []
    step = duration / frames
    return [round(step * (i + 0.5), 3) for i in range(frames)]


def label(t: float) -> str:
    return f"{int(t)}s"


def _duration(video: Path) -> float:
    info = shell.probe(video)
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        raise ValueError(f"{video.name} has no frames: there is no video track")
    seconds = info.get("format", {}).get("duration")
    if seconds is None:
        seconds = max(float(s.get("duration", 0.0)) for s in info["streams"])
    return float(seconds)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):            # Pillow without FreeType: the fixed bitmap font
        return ImageFont.load_default()


def contact_sheet(video: Path, dest: Path, frames: int = 15) -> Path:
    """Frames from the middle of `frames` equal slices of `video`, on one JPEG.

    A cell whose time comes after the last frame (a picture that ends before its sound) is
    left black. Raises ValueError when the file has no frames at all."""
    video, dest = Path(video), Path(dest)
    times = frame_times(_duration(video), frames)
    cells: list[Image.Image | None] = [None] * len(times)
    if times:
        step = times[1] - times[0] if len(times) > 1 else times[0] * 2
        # The sampler ticks at exactly the cell times and yields the first frame at or after
        # each one, in a single forward read. A frame fills every cell it is the answer for.
        for t, rgb in iter_sampled_frames(video, times[0], times[-1] + step / 2, fps=1.0 / step):
            picture = None
            for i, due in enumerate(times):
                if cells[i] is None and due <= t + 0.001:
                    if picture is None:
                        picture = ImageOps.pad(Image.fromarray(rgb), (CELL_W, CELL_H),
                                               color=(0, 0, 0))
                    cells[i] = picture
    if not any(cells):
        raise ValueError(f"{video.name} has no frames that could be read")

    grid = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    font = _font(16)
    for i, (due, picture) in enumerate(zip(times, cells)):
        x, y = (i % COLS) * CELL_W, (i // COLS) * CELL_H
        if picture is not None:
            grid.paste(picture, (x, y))
        text = label(due)
        width = int(draw.textlength(text, font=font)) + 12
        draw.rectangle((x, y, x + width - 1, y + TAB_H - 1), fill=(0, 0, 0))
        draw.text((x + 6, y + TAB_H // 2), text, fill=(255, 255, 255), font=font, anchor="lm")
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid.save(dest, "JPEG", quality=QUALITY)
    return dest
