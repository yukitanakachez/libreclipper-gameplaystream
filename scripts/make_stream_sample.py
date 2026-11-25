"""Build the stand-in sample stream: agents/gameplay-clipper/tests/sample-stream.mp4.

    uv run python scripts/make_stream_sample.py [--work DIR] [--out FILE]

Five minutes that look enough like a gameplay stream to run the whole engine on: a game
drawn by this script fills the frame, and a talking face sits in a bordered camera box in
the top-left corner. Every part is listed with its licence in
agents/gameplay-clipper/tests/SOURCES.md. Only the face footage is downloaded (public
domain, NASA); the game picture and the game sound are made here, from a fixed seed.

The parts go in a scratch folder and are never committed. One ffmpeg command puts it all
together: it scales the game to 1280x720, lays the face box over it, mixes the speech over
the game sound with the game 18 dB lower, applies the loud moments, freezes and silences
the dead-air stretch, and encodes small enough to commit.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clipper import shell  # noqa: E402

DEFAULT_OUT = ROOT / "agents" / "gameplay-clipper" / "tests" / "sample-stream.mp4"

# --- What the finished file is ---------------------------------------------------------
DURATION = 300.0
FPS = 30
WIDTH, HEIGHT = 1280, 720
RATE = 48000

# The camera box: a 320x256 picture (a quarter of the frame's width), a 4 px border around
# it, 24 px from the top-left corner. It is taller than 16:9 because the face finder only
# finds faces at least ~200 px tall, and the whole face must sit inside the box.
CAM_W, CAM_H, BORDER, MARGIN = 320, 256, 4, 24
BORDER_COLOUR = "0xF2F2F2"

# Loud moments (start, end) and the dead-air stretch, in stream seconds.
LOUD = [(60.0, 62.0), (135.0, 137.0), (250.0, 252.0)]
DEAD_AIR = (200.0, 204.0)

SPEECH_LUFS = -20.0   # the speech is brought to this loudness
GAME_UNDER_DB = 18.0  # the game's steady sound sits this far below the speech
LOUD_BOOST_DB = 10.0  # the speech is raised this much during a loud moment

# --- The face (public domain, NASA) ----------------------------------------------------
FACE_ID = "Artemis Media Resource-Interview Soundbites"
FACE_URL = ("https://images-assets.nasa.gov/video/"
            "Artemis%20Media%20Resource-Interview%20Soundbites/"
            "Artemis%20Media%20Resource-Interview%20Soundbites~large.mp4")
FACE_SIZE = 252_227_642  # bytes, as served on 2026-11-23

# Each speaker's answers, in source seconds. Every stretch is an answer with 1.8 s of the
# pause around it, stopping short of the black frames and title cards between answers.
# Each speaker's crop is centred on the middle of their face in the 1920x1080 source and
# scaled by CAM_SCALE, so the face is about 170 px tall in the box.
CAM_SCALE = 0.55


def crop_at(cx: float, cy: float) -> tuple[int, int, int, int]:
    w, h = round(CAM_W / CAM_SCALE / 2) * 2, round(CAM_H / CAM_SCALE / 2) * 2
    return (int(min(1920 - w, max(0, cx - w / 2))), int(min(1080 - h, max(0, cy - h / 2))), w, h)


BRIDENSTINE = {
    "crop": crop_at(762, 445),
    "spans": [(16.32, 36.04), (37.92, 56.99), (58.74, 69.28), (70.91, 89.19),
              (90.84, 111.90), (113.45, 129.99), (131.46, 161.06), (163.05, 181.04)],
}
WISEMAN = {
    "crop": crop_at(977, 397),
    "spans": [(194.64, 222.35), (223.83, 250.32), (252.02, 263.39), (264.83, 287.82),
              (289.35, 306.55)],
}


def face_plan() -> list[tuple[float, float, tuple[int, int, int, int]]]:
    """(source start, source end, crop) pieces that add up to exactly DURATION seconds:
    every Bridenstine answer, every Wiseman answer, then Bridenstine again from the top."""
    plan: list[tuple[float, float, tuple[int, int, int, int]]] = []
    total = 0.0
    for speaker in (BRIDENSTINE, WISEMAN, BRIDENSTINE):
        for a, b in speaker["spans"]:
            left = DURATION - total
            if left <= 0:
                return plan
            b = min(b, a + left)
            plan.append((a, b, speaker["crop"]))
            total += b - a
    if total < DURATION - 1e-6:
        raise SystemExit("face footage too short for the plan")
    return plan


# --- Downloads -------------------------------------------------------------------------
def download(url: str, dest: Path, size: int | None = None) -> Path:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print(f"have {dest.name}")
        return dest
    print(f"downloading {url}")
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "stream-sample-builder/1.0"})
    with urllib.request.urlopen(req) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    if size is not None and part.stat().st_size != size:
        raise SystemExit(f"{dest.name}: got {part.stat().st_size} bytes, expected {size}")
    part.replace(dest)
    return dest


# --- The game: a side-scrolling space shooter, drawn here ------------------------------
GW, GH = 640, 360  # drawn at half size, scaled up by ffmpeg
SKY_TOP = np.array([10, 12, 40], dtype=np.float32)
SKY_BOTTOM = np.array([60, 20, 70], dtype=np.float32)


class Game:
    """Every frame is a pure function of the frame number and a fixed seed."""

    def __init__(self, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.stars = [(rng.uniform(0, GW), rng.uniform(0, GH * 0.75), layer)
                      for layer in (1, 2, 3) for _ in range(110)]
        grad = np.linspace(0, 1, GH, dtype=np.float32)[:, None]
        sky = SKY_TOP * (1 - grad) + SKY_BOTTOM * grad
        self.sky = np.repeat(sky[:, None, :], GW, axis=1).astype(np.uint8)
        self.ridge_phase = rng.uniform(0, 6.28, size=3)
        # Enemy waves: (spawn time, lane y, speed px/s, kind)
        self.enemies = []
        t = 1.0
        while t < DURATION:
            for k in range(int(rng.integers(3, 6))):
                self.enemies.append((t + k * 0.45, float(rng.uniform(70, 260)),
                                     float(rng.uniform(110, 170)), int(rng.integers(0, 3))))
            t += float(rng.uniform(1.5, 3.0))
        self.shots = np.arange(0.5, DURATION, 0.35)

    def ship_y(self, t: float) -> float:
        return 200 + 55 * math.sin(t * 0.9) + 20 * math.sin(t * 2.3)

    def enemy_pos(self, e, t: float) -> tuple[float, float] | None:
        spawn, lane, speed, kind = e
        age = t - spawn
        if age < 0:
            return None
        x = GW + 20 - speed * age
        y = lane + 25 * math.sin(age * (1.5 + kind))
        return (x, y) if x > -20 else None

    def kill_time(self, e) -> float:
        """Each enemy is shot when it is 62% of the way across the screen from the left."""
        spawn, _, speed, _ = e
        return spawn + (GW + 20 - GW * 0.62) / speed

    def frame(self, n: int) -> bytes:
        t = n / FPS
        img = Image.fromarray(self.sky.copy())
        d = ImageDraw.Draw(img)
        # Stars in three layers, scrolling at different speeds.
        for x, y, layer in self.stars:
            sx = int((x - t * 30 * layer ** 1.6) % GW)
            sy = int(y)
            c = 90 + 50 * layer
            d.rectangle([sx, sy, sx + layer - 1, sy + layer - 1], fill=(c, c, min(255, c + 30)))
        # Two ridges of hills at the bottom.
        for depth, (col, base, amp, speed) in enumerate(
                [((40, 30, 80), 290, 30, 60), ((25, 60, 55), 320, 22, 140)]):
            pts = [(0, GH)]
            for x in range(0, GW + 16, 16):
                u = (x + t * speed) / 90.0
                pts.append((x, base - amp * (0.6 * math.sin(u + self.ridge_phase[depth])
                                             + 0.4 * math.sin(2.7 * u))))
            pts.append((GW, GH))
            d.polygon(pts, fill=col)
        # Player ship.
        sy = self.ship_y(t)
        sx = 230 + 25 * math.sin(t * 0.5)
        d.polygon([(sx + 28, sy), (sx - 12, sy - 12), (sx - 4, sy), (sx - 12, sy + 12)],
                  fill=(120, 220, 255))
        flame = 6 + 4 * ((n // 2) % 2)
        d.polygon([(sx - 6, sy - 4), (sx - 6 - flame, sy), (sx - 6, sy + 4)], fill=(255, 170, 40))
        # Bullets.
        for s in self.shots[(self.shots <= t) & (self.shots > t - 1.6)]:
            bx = 230 + 25 * math.sin(s * 0.5) + 30 + (t - s) * 420
            by = self.ship_y(s)
            if bx < GW:
                d.rectangle([bx, by - 1, bx + 10, by + 1], fill=(255, 255, 140))
        # Enemies, and their explosions once shot.
        for e in self.enemies:
            if e[0] > t:
                continue
            kt = self.kill_time(e)
            if t < kt:
                pos = self.enemy_pos(e, t)
                if pos is None:
                    continue
                x, y = pos
                col = [(255, 90, 90), (255, 160, 60), (200, 100, 255)][e[3]]
                if e[3] == 0:
                    d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col)
                elif e[3] == 1:
                    d.polygon([(x - 11, y), (x, y - 11), (x + 11, y), (x, y + 11)], fill=col)
                else:
                    d.rectangle([x - 8, y - 8, x + 8, y + 8], fill=col)
            elif t < kt + 0.6:
                pos = self.enemy_pos(e, kt)
                if pos is None:
                    continue
                x, y = pos
                r = 4 + 40 * (t - kt)
                a = 1 - (t - kt) / 0.6
                d.ellipse([x - r, y - r, x + r, y + r], outline=(255, int(200 * a), 60), width=3)
        # Big blasts at the loud moments: a growing fireball and a white flash.
        for a, b in LOUD:
            if a <= t < b + 0.5:
                k = (t - a) / (b - a + 0.5)
                r = 30 + 260 * k
                cx, cy = GW * 0.62, GH * 0.45
                d.ellipse([cx - r, cy - r, cx + r, cy + r],
                          fill=(255, int(220 - 150 * k), int(80 * (1 - k))))
                r2 = r * (0.55 + 0.1 * math.sin(t * 40))
                d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=(255, 245, 160))
                if t < a + 0.25:
                    d.rectangle([0, 0, GW, GH], fill=(255, 255, 255))
        # HUD: lives and a score bar, top right (the camera box covers the top left).
        for i in range(3):
            x0 = GW - 24 - i * 18
            d.polygon([(x0 + 12, 18), (x0, 12), (x0, 24)], fill=(120, 220, 255))
        filled = int(150 * ((t / DURATION) % 1.0))
        d.rectangle([GW - 180, 34, GW - 30, 40], outline=(200, 200, 220))
        d.rectangle([GW - 180, 34, GW - 180 + filled, 40], fill=(215, 255, 62))
        return img.tobytes()


# --- The game's sound, made here --------------------------------------------------------
def db(x: float) -> float:
    return 10 ** (x / 20)


def game_sound(game: Game) -> np.ndarray:
    """Stereo float32 at RATE: an engine drone, a blip per shot, a pop per kill, and a
    big boom at each loud moment. The steady sound sits near SPEECH_LUFS - GAME_UNDER_DB."""
    rng = np.random.default_rng(11)
    n = int(DURATION * RATE)
    t = np.arange(n, dtype=np.float64) / RATE
    # Drone: two low tones with slow wobble, plus soft filtered noise.
    drone = 0.6 * np.sin(2 * np.pi * 55 * t + 0.3 * np.sin(2 * np.pi * 0.2 * t))
    drone += 0.3 * np.sin(2 * np.pi * 82.5 * t)
    noise = rng.standard_normal(n)
    kernel = np.ones(64) / 64
    drone += 0.8 * np.convolve(noise, kernel, mode="same")
    drone *= db(SPEECH_LUFS - GAME_UNDER_DB) / np.sqrt(np.mean(drone ** 2))
    out = drone

    def add(at: float, clip: np.ndarray) -> None:
        i = int(at * RATE)
        j = min(n, i + len(clip))
        if i < n:
            out[i:j] += clip[: j - i]

    # Shots: a short falling chirp.
    k = np.arange(int(0.08 * RATE)) / RATE
    chirp = np.sin(2 * np.pi * (1800 * k - 6000 * k ** 2)) * np.exp(-k * 40) * db(-34)
    for s in game.shots:
        add(float(s), chirp)
    # Kills: a short noise pop.
    k = np.arange(int(0.25 * RATE)) / RATE
    for e in game.enemies:
        kt = game.kill_time(e)
        if kt < DURATION:
            pop = rng.standard_normal(len(k)) * np.exp(-k * 18) * db(-30)
            add(kt, np.convolve(pop, np.ones(8) / 8, mode="same"))
    # Loud moments: a 2 s boom, low tone plus rumbling noise.
    for a, b in LOUD:
        k = np.arange(int((b - a) * RATE)) / RATE
        env = np.minimum(1.0, k / 0.02) * np.exp(-k * 0.6)
        rumble = np.convolve(rng.standard_normal(len(k)), np.ones(24) / 24, mode="same")
        rumble /= np.sqrt(np.mean(rumble ** 2))
        boom = (0.6 * np.sin(2 * np.pi * 50 * k) + 0.7 * rumble) * env
        boom *= db(-7) / np.sqrt(np.mean(boom ** 2))
        add(a, boom)
    left = out * (1 + 0.1 * np.sin(2 * np.pi * 0.07 * t))
    right = out * (1 - 0.1 * np.sin(2 * np.pi * 0.07 * t))
    return np.clip(np.stack([left, right], axis=1), -1, 1).astype(np.float32)


def write_wav(path: Path, stereo: np.ndarray) -> None:
    pcm = (stereo * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


# --- The one ffmpeg command ------------------------------------------------------------
def build_command(face: Path, game_wav: Path, out: Path) -> list[str]:
    plan = face_plan()
    cmd = [shell.ffmpeg(), "-v", "error", "-stats", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{GW}x{GH}", "-r", str(FPS),
           "-i", "pipe:0", "-i", str(game_wav)]
    for a, b, _ in plan:
        cmd += ["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(face)]
    parts = []
    for i, (_, _, (x, y, w, h)) in enumerate(plan):
        k = i + 2
        parts.append(f"[{k}:v]crop={w}:{h}:{x}:{y},scale={CAM_W}:{CAM_H},fps={FPS},"
                     f"setsar=1,format=yuv420p[v{i}];"
                     f"[{k}:a]aresample={RATE},aformat=channel_layouts=stereo[a{i}]")
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(plan)))
    loud_on = "+".join(f"between(t,{a},{b})" for a, b in LOUD)
    da, dz = DEAD_AIR
    first, last = int(da * FPS), int(dz * FPS) - 1
    graph = ";".join(parts) + ";" + (
        f"{pairs}concat=n={len(plan)}:v=1:a=1[cam][speech0];"
        f"[cam]pad={CAM_W + 2 * BORDER}:{CAM_H + 2 * BORDER}:{BORDER}:{BORDER}:"
        f"color={BORDER_COLOUR}[box];"
        f"[0:v]scale={WIDTH}:{HEIGHT}:flags=bicubic,setsar=1,format=yuv420p[game];"
        f"[game][box]overlay={MARGIN}:{MARGIN}:eof_action=repeat,"
        f"trim=duration={DURATION},setpts=PTS-STARTPTS,split[pa][pb];"
        f"[pa][pb]freezeframes=first={first}:last={last}:replace={first - 1}[vout];"
        # Volume switches act on whole audio frames, so frames are cut to 10 ms first.
        f"[speech0]loudnorm=I={SPEECH_LUFS}:TP=-2:LRA=7,aresample={RATE},"
        f"asetnsamples=n={RATE // 100}:p=0,volume={LOUD_BOOST_DB}dB:enable='{loud_on}'[speech];"
        f"[1:a]aformat=channel_layouts=stereo[gamea];"
        f"[speech][gamea]amix=inputs=2:duration=longest:normalize=0,"
        f"alimiter=limit=0.95:level=disabled:latency=1,"
        f"asetnsamples=n={RATE // 100}:p=0,volume=0:enable='between(t,{da},{dz})',"
        f"atrim=duration={DURATION},asetpts=PTS-STARTPTS[aout]")
    cmd += ["-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-b:v", "430k", "-maxrate", "560k", "-bufsize", "1120k",
            "-g", str(FPS), "-keyint_min", str(FPS), "-sc_threshold", "0", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "96k", "-ar", str(RATE), "-ac", "2",
            "-movflags", "+faststart", "-t", str(DURATION), str(out)]
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path,
                    default=Path(tempfile.gettempdir()) / "stream-sample-parts",
                    help="scratch folder for the downloaded and generated parts")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    face = download(FACE_URL, args.work / "artemis-soundbites.mp4", FACE_SIZE)
    game = Game()
    wav = args.work / "game-sound.wav"
    print("making the game sound")
    write_wav(wav, game_sound(game))

    cmd = build_command(face, wav, args.out)
    print("rendering")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for n in range(int(DURATION * FPS)):
            proc.stdin.write(game.frame(n))
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed ({proc.returncode})")

    size = args.out.stat().st_size
    print(f"\nwrote {args.out} ({size / 1e6:.1f} MB)")
    print("loud moments: " + ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in LOUD))
    print(f"dead air: {DEAD_AIR[0]:.0f}-{DEAD_AIR[1]:.0f} s")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
