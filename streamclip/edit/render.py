"""The render graph and the encode: one finished 1080x1920 clip from one stretch of source.

Order of operations, the podcast clipper's: lay out the panels, drop the cut frames, draw
the title, tag and captions (all timed to the cut clip), then speed everything up together.
Sounds are mixed in after the speed-up, so they play at their natural speed.

Three clocks: shots and `keep` are in source-clip time, overlays, captions and `Placed.at`
in cut time, and the file that comes out plays in finished time (cut time / speed).

ffmpeg's inputs are, in order: the source, each overlay image, each sound.
"""

from __future__ import annotations

import os
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from clipper import shell
from clipper.render import _crop, _loudnorm_filter, _pace_audio, parse_loudnorm_json
from streamclip import settings
from streamclip.errors import NoVideo, StageError
from streamclip.schema import Layout, Placed, Shot, TextAssets

ZOOM_SECONDS = 0.7          # how long the face panel stays on its tighter crop
SEAM_PX = 6                 # the accent line where the two panels meet
X264_PRESET = "fast"        # the default; STREAMCLIP_X264_PRESET overrides it (Starwish: ultrafast)
X264_CRF = 20
_STEREO = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


def encode_options() -> list[str]:
    """The picture encoder's settings: the preset (STREAMCLIP_X264_PRESET, read when a clip is
    rendered) and as many threads as this job may use."""
    preset = os.environ.get("STREAMCLIP_X264_PRESET", "").strip() or X264_PRESET
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(X264_CRF), "-pix_fmt", "yuv420p",
            "-threads", str(settings.processors())]


def segments(layout: Layout, zoom_t: float | None) -> list[Shot]:
    """The layout's shots, with the one that holds `zoom_t` split into before, zoomed and
    after. The zoomed part uses the shot's `zoom` in place of its face crop. A game shot, a
    shot with no zoom crop, or no zoom time changes nothing."""
    out: list[Shot] = []
    for shot in layout.shots:
        zoomable = (zoom_t is not None and shot.kind != "game" and shot.zoom is not None
                    and shot.start <= zoom_t < shot.end)
        if not zoomable:
            out.append(shot)
            continue
        until = min(round(zoom_t + ZOOM_SECONDS, 6), shot.end)
        pieces = [replace(shot, end=zoom_t),
                  replace(shot, start=zoom_t, end=until, face=shot.zoom),
                  replace(shot, start=until)]
        out += [p for p in pieces if p.end > p.start]
    return out


def _fonts_dir() -> Path:
    try:
        from streamclip.text import FONTS_DIR
        return Path(FONTS_DIR)
    except ImportError:
        return Path(__file__).resolve().parent.parent / "assets" / "fonts"


# A game crop this much wider than its panel is shown whole instead of stretched to fill it.
WIDE = 1.05


def _game_panel(trim: str, rect, src_w: int, src_h: int, w: int, h: int, label: str) -> str:
    """The game panel. A crop with the panel's shape fills it. A wider one (a menu that needs
    the whole screen to read) is shown whole across the panel, over a dimmed blur of itself."""
    shape = (rect.w * src_w) / max(rect.h * src_h, 1e-6)
    if shape <= (w / h) * WIDE:
        return f"{trim},{_crop(rect, src_w, src_h)},scale={w}:{h},setsar=1[{label}]"
    return (f"{trim},{_crop(rect, src_w, src_h)},split[{label}f][{label}g];"
            f"[{label}g]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
            f"boxblur=24:2,eq=brightness=-0.12[{label}bg];"
            f"[{label}f]scale={w}:-2[{label}fg];"
            f"[{label}bg][{label}fg]overlay=(W-w)/2:(H-h)/2,setsar=1[{label}]")


def _video_graph(shots: list[Shot], keep: list[tuple[float, float]], text: TextAssets,
                 speed: float, src_w: int, src_h: int, fps: float, length: float) -> list[str]:
    w, h = settings.OUT_W, settings.OUT_H
    parts: list[str] = []
    labels: list[str] = []
    for i, shot in enumerate(shots):
        start, end = max(shot.start, 0.0), min(shot.end, length)
        if end <= start:
            continue
        trim = f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS"
        if shot.kind == "split" and shot.face is not None and shot.game is not None:
            parts.append(f"{trim},{_crop(shot.face, src_w, src_h)},"
                         f"scale={w}:{settings.FACE_H},setsar=1[v{i}a]")
            parts.append(_game_panel(trim, shot.game, src_w, src_h, w, settings.GAME_H, f"v{i}b"))
            seam = settings.FACE_H - SEAM_PX // 2
            parts.append(f"[v{i}a][v{i}b]vstack=inputs=2,"
                         f"drawbox=x=0:y={seam}:w={w}:h={SEAM_PX}:"
                         f"color=0x{settings.ACCENT.lstrip('#')}:t=fill,setsar=1[v{i}]")
        else:
            # One crop fills the frame: the face, the game, or whichever a broken split has.
            crop = (shot.game or shot.face) if shot.kind == "game" else (shot.face or shot.game)
            if crop is None:
                raise StageError(f"The layout has a {shot.kind} shot with nothing to show.")
            parts.append(f"{trim},{_crop(crop, src_w, src_h)},scale={w}:{h},setsar=1[v{i}]")
        labels.append(f"[v{i}]")
    if not labels:
        raise StageError("The layout has no shots inside the clip.")
    # A fixed frame rate here, so the cut below counts frames on the grid `keep` was landed on.
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0,fps={fps}[vcat]")

    current = "[vcat]"
    if keep:
        half = 0.5 / fps
        expr = "+".join(f"gte(t\\,{a - half:.4f})*lt(t\\,{b - half:.4f})" for a, b in keep)
        parts.append(f"{current}select='{expr}',setpts=N/({fps}*TB)[vcut]")
        current = "[vcut]"
    for i, overlay in enumerate(text.overlays):
        when = ("" if overlay.end is None else
                f":enable='between(t,{overlay.start:.3f},{overlay.end:.3f})'")
        if overlay.end is None and overlay.start > 0:
            when = f":enable='gte(t,{overlay.start:.3f})'"
        parts.append(f"{current}[{i + 1}:v]overlay={overlay.x}:{overlay.y}{when}[vo{i}]")
        current = f"[vo{i}]"
    last: list[str] = []
    if text.ass:
        last.append(f"ass={shell.filter_path(text.ass)}:fontsdir={shell.filter_path(_fonts_dir())}")
    if speed != 1.0:
        # The rate is set here and not only by the output's -r flag, which kept extra
        # frames and left the picture longer than the sound.
        last += [f"setpts=PTS/{speed:.3f}", f"fps={min(60.0, fps):g}"]
    parts.append(f"{current}{','.join(last) or 'null'}[vout]")
    return parts


def _audio_graph(keep: list[tuple[float, float]], sounds: list[Placed], speed: float,
                 first_sound_input: int, length: float, measured: dict | None,
                 measure: bool) -> list[str]:
    parts, voice = _pace_audio("[0:a]", keep or [(0.0, length)], speed)
    parts.append(f"{voice}{_STEREO}[voice]")
    current = "[voice]"
    if sounds:
        labels = []
        for i, placed in enumerate(sounds):
            delay = max(0, round(placed.at / speed * 1000))
            parts.append(f"[{first_sound_input + i}:a]{_STEREO},volume={placed.gain_db:.1f}dB,"
                         f"adelay={delay}|{delay}[s{i}]")
            labels.append(f"[s{i}]")
        # duration=first: a sound never makes the clip longer than its picture.
        parts.append(f"[voice]{''.join(labels)}amix=inputs={len(sounds) + 1}:duration=first:"
                     "normalize=0[mixed]")
        current = "[mixed]"
    # loudnorm puts out 192 kHz, which AAC will not take.
    parts.append(f"{current}{_loudnorm_filter(measured, measure)},aresample=48000[aout]")
    return parts


def build_graph(shots: list[Shot], keep: list[tuple[float, float]], text: TextAssets,
                sounds: list[Placed], speed: float, src_w: int, src_h: int, fps: float,
                length: float, measured: dict | None = None, measure: bool = False) -> str:
    """The whole filter graph, ending in [vout] and [aout]. `length` is the source clip's
    length in seconds. With `measure`, only the sound half is built (ending in [aout]) and
    its loudness is printed, not corrected: that is the cheap first pass."""
    sound_half = _audio_graph(keep, sounds, speed, 1 + len(text.overlays), length,
                              measured, measure)
    if measure:
        return ";".join(sound_half)
    return ";".join(_video_graph(shots, keep, text, speed, src_w, src_h, fps, length) + sound_half)


def _probe(source: Path) -> tuple[int, int, float, float]:
    data = shell.probe(source)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise NoVideo(f"{Path(source).name} has no picture, only sound.")
    if not any(s["codec_type"] == "audio" for s in data["streams"]):
        raise StageError(f"{Path(source).name} has no sound.")
    rate = video.get("avg_frame_rate", "0/0")
    if rate in ("", "0/0"):
        rate = video.get("r_frame_rate", "30/1")
    fps = float(Fraction(rate)) or 30.0
    length = float(video.get("duration") or data["format"].get("duration") or 0.0)
    return int(video["width"]), int(video["height"]), fps, length


def render_clip(source: Path, layout: Layout, keep: list[tuple[float, float]],
                zoom_t: float | None, text: TextAssets, sounds: list[Placed], speed: float,
                dest: Path) -> Path:
    """Encode one finished clip to `dest`. The file appears only when it is complete."""
    source, dest = Path(source), Path(dest)
    src_w, src_h, fps, length = _probe(source)
    keep = [(max(0.0, a), min(b, length)) for a, b in keep if min(b, length) > max(0.0, a)]
    keep = keep or [(0.0, length)]
    sounds = [s for s in sounds if Path(s.path).is_file()]      # a missing sound costs the sound, not the clip
    shots = segments(layout, zoom_t)

    inputs = [shell.ffmpeg(), "-v", "info", "-nostdin", "-y", "-i", str(source)]
    for overlay in text.overlays:
        inputs += ["-i", overlay.path]
    for placed in sounds:
        inputs += ["-i", placed.path]

    def graph(**how) -> str:
        return build_graph(shots, keep, text, sounds, speed, src_w, src_h, fps, length, **how)

    # Pass 1: measure the mixed sound's loudness. Sound only, so it is cheap.
    measured = parse_loudnorm_json(shell.run_capture_stderr(
        inputs + ["-filter_complex", graph(measure=True), "-map", "[aout]", "-f", "null", "-"]))

    # Pass 2: the encode, with that measurement applied.
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.stem + ".partial.mp4")
    try:
        shell.run(inputs + [
            "-filter_complex", graph(measured=measured), "-map", "[vout]", "-map", "[aout]",
            *encode_options(), "-r", f"{min(60.0, fps):g}",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", str(partial)])
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest
