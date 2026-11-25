"""Measure the stream clipper on real streams, for tuning (Lane N).

    python scripts/stream_tune.py <jobs folder>
        One row per job folder under it, and a total: viewer-clip hit rate, judge keep
        rate, spread, extras, stage times and dead air. Reads only what the jobs hold.

    python scripts/stream_tune.py <jobs folder> --rescore [--set curve.WEIGHTS.voice=0.4 ...]
        The same table with each shortlist built again from the job's saved sound, under
        the constants as they are now (and any --set). Nothing in the jobs is changed.
        Adds the hit rate with viewer clips withheld from the curve: clips are both a
        signal and the answer key, so only the withheld figure is a fair test.

    python scripts/stream_tune.py run <link or file> --job <folder> [--clips N]
        The engine's find, judge and make, timed step by step into <job>/tuning.json.

    python scripts/stream_tune.py refind <job folder> [...]
        Build the curve, the shortlist and the candidates again from the saved sound.

Job folders hold other people's streams. They stay outside the repo and are deleted when
tuning ends; only the numbers this prints are kept.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HALF_HOUR = 30 * 60
KEEP_SCORE = 7                      # the judge kept it and gave it this or more
TIMINGS = "tuning.json"
CACHE = "tuning-cache.npz"
STAGES = ("find", "judge", "make")
EXTRAS = ("chat", "viewer_clips", "avatar")


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- the measurements ----------------------------------------------------------------------

def hits(clips: list[dict], windows: list[tuple[float, float]]) -> tuple[float, float]:
    """(views of the clips whose middle lies inside a shortlisted window, all views).
    A clip nobody watched still counts as one view, so it is not ignored."""
    inside = total = 0.0
    for clip in clips:
        middle = float(clip["offset"]) + float(clip.get("duration") or 0.0) / 2
        views = max(1.0, float(clip.get("views") or 0))
        total += views
        if any(a <= middle <= b for a, b in windows):
            inside += views
    return inside, total


def busiest_share(peaks: list[float], span: float = HALF_HOUR) -> float | None:
    """The share of the shortlist that falls in its busiest `span` seconds."""
    peaks = sorted(peaks)
    if not peaks:
        return None
    most = max(sum(1 for q in peaks[i:] if q - p < span) for i, p in enumerate(peaks))
    return most / len(peaks)


def measure(job: Path, hotspots: list[dict] | None = None) -> dict:
    """One stream's row. `hotspots` replaces the job's own shortlist (for --rescore)."""
    job = Path(job)
    stream = _read(job / "stream.json") or {}
    if hotspots is None:
        hotspots = (_read(job / "hotspots.json") or {}).get("hotspots") or []
    windows = [(float(h["start"]), float(h["end"])) for h in hotspots]

    clips = _read(job / "source" / "viewer_clips.json")
    hit_views, clip_views = hits(clips, windows) if clips else (0.0, 0.0)

    judged = (_read(job / "judge_raw.json") or {}).get("candidates")
    kept = None
    if isinstance(judged, list):
        kept = sum(1 for row in judged if isinstance(row, dict) and row.get("keep") is True
                   and float(row.get("score") or 0) >= KEEP_SCORE)

    timings = _read(job / TIMINGS) or {}
    stages = {name: float(timings["stages"][name]) for name in STAGES
              if name in (timings.get("stages") or {})}
    made = (_read(job / "out" / "clips.json") or {}).get("clips") or []
    noted = stream.get("extras") or {}
    return {
        "name": job.name,
        "duration": float(stream.get("duration") or 0.0),
        "candidates": len(hotspots),
        "viewer_clips": len(clips or []),
        "hit_views": hit_views, "clip_views": clip_views,
        "hit_rate": hit_views / clip_views if clip_views else None,
        "kept": kept,
        "keep_rate": kept / len(hotspots) if kept is not None and hotspots else None,
        "spread": busiest_share([float(h["peak"]) for h in hotspots]),
        "extras": {name: noted.get(name) == "ok" for name in EXTRAS},
        "chat_seconds": timings.get("chat_seconds"),
        "stages": stages,
        "steps": dict(timings.get("steps") or {}),
        "total_seconds": sum(stages.values()) if stages else None,
        "dead_air": [(int(c["rank"]), float(c["seconds_cut"]), int(c["cuts"])) for c in made],
    }


def total(rows: list[dict]) -> dict:
    """The set as one row: every viewer's view and every candidate counts once."""
    views = sum(r["clip_views"] for r in rows)
    judged = [r for r in rows if r["kept"] is not None]
    candidates = sum(r["candidates"] for r in judged)
    timed = [r["total_seconds"] for r in rows if r["total_seconds"] is not None]
    blind_views = sum(r["clip_views"] for r in rows if r.get("blind_hit_views") is not None)
    return {
        "name": "total", "duration": sum(r["duration"] for r in rows),
        "candidates": sum(r["candidates"] for r in rows),
        "viewer_clips": sum(r["viewer_clips"] for r in rows),
        "hit_rate": sum(r["hit_views"] for r in rows) / views if views else None,
        "blind_hit_rate": (sum(r.get("blind_hit_views") or 0.0 for r in rows) / blind_views
                           if blind_views else None),
        "keep_rate": sum(r["kept"] for r in judged) / candidates if candidates else None,
        "spread": None, "extras": None, "chat_seconds": None,
        "stages": {name: sum(r["stages"].get(name, 0.0) for r in rows) for name in STAGES} if timed else {},
        "total_seconds": sum(timed) if timed else None,
        "dead_air": [cut for r in rows for cut in r["dead_air"]],
    }


# --- the table -----------------------------------------------------------------------------

def cell(value, kind: str = "share") -> str:
    """A table cell. Nothing measured is an empty cell, never a zero."""
    if value is None:
        return ""
    if kind == "share":
        return f"{100 * value:.0f}%"
    if kind == "seconds":
        return f"{value:.0f} s"
    return str(value)


def _extras_cell(row: dict) -> str:
    if row["extras"] is None:
        return ""
    words = [f"{name.replace('_', ' ')} {'yes' if ok else 'no'}" for name, ok in row["extras"].items()]
    if row["chat_seconds"] is not None:
        words.append(f"chat fetch {row['chat_seconds']:.0f} s")
    return ", ".join(words)


def _dead_air_cell(row: dict) -> str:
    cuts = row["dead_air"]
    if not cuts:
        return ""
    if row["name"] == "total":
        return f"{sum(s for _, s, _ in cuts):.1f} s in {sum(n for _, _, n in cuts)} cuts, {len(cuts)} clips"
    return "; ".join(f"{rank:02d}: {secs:.1f} s/{n}" for rank, secs, n in cuts)


def table(rows: list[dict], blind: bool = False) -> str:
    head = ["stream", "length", "shortlist", "viewer clips", "hit rate"]
    head += ["hit rate, clips withheld"] if blind else []
    head += ["judge keep rate", "busiest 30 min", "extras", "find", "judge", "make", "total",
             "dead air (clip: seconds cut/cuts)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [row["name"], f"{row['duration'] / 3600:.2f} h", str(row["candidates"]),
                 str(row["viewer_clips"]), cell(row["hit_rate"])]
        cells += [cell(row.get("blind_hit_rate"))] if blind else []
        cells += [cell(row["keep_rate"]), cell(row["spread"]), _extras_cell(row)]
        cells += [cell(row["stages"].get(name), "seconds") for name in STAGES]
        cells += [cell(row["total_seconds"], "seconds"), _dead_air_cell(row)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def jobs_under(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if (p / "stream.json").is_file() and (p / "hotspots.json").is_file())


# --- building the shortlist again from the saved sound -------------------------------------

def apply_settings(pairs: list[str]) -> None:
    """`curve.WEIGHTS.voice=0.4`, `loudness.LOUD_FULL_DB=10`, `shortlist.SMOOTH=7`, ...:
    set a named constant for this run only. The files are not touched."""
    from streamclip.edit import deadair
    from streamclip.signals import curve, extras, loudness, shortlist
    modules = {"curve": curve, "loudness": loudness, "extras": extras, "shortlist": shortlist,
               "deadair": deadair}
    for pair in pairs:
        try:
            path, text = pair.split("=", 1)
            module, name, *key = path.split(".")
            target, value = modules[module], float(text)
            current = getattr(target, name)
            if key:
                if key[0] not in current:
                    raise KeyError(key[0])
                current[key[0]] = value
            else:
                setattr(target, name, int(value) if isinstance(current, int) else value)
        except (ValueError, KeyError, AttributeError):
            raise SystemExit(f"no such constant: {pair!r}") from None


def sound_scores(job: Path):
    """The job's per-second levels and classifier scores, worked out once and kept beside it.
    Neither depends on a tuned constant, so every trial reuses them."""
    from streamclip.signals import events, loudness
    job = Path(job)
    cache = job / CACHE
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as kept:
            return kept["db"], events.VoiceTrack(kept["voice"], kept["music"], kept["speech"],
                                                 [str(s) for s in kept["label"]])
    duration = float(_read(job / "stream.json")["duration"])
    audio = job / "source" / "audio.m4a"
    db = loudness.per_second_db(audio, duration)
    track = events.voice_events(audio, duration)
    _save_scores(cache, db, track)
    return db, track


def _save_scores(cache: Path, db, track) -> None:
    np.savez_compressed(cache, db=db, voice=track.voice, music=track.music, speech=track.speech,
                        label=np.asarray(track.label, dtype=str))


@contextlib.contextmanager
def saved_sound(job: Path):
    """Inside, the engine's curve reads this job's kept scores and not the sound itself."""
    from streamclip.signals import events, loudness
    db, track = sound_scores(job)
    real = loudness.per_second_db, events.voice_events
    loudness.per_second_db = lambda *_a, **_k: db
    events.voice_events = lambda *_a, **_k: track
    try:
        yield
    finally:
        loudness.per_second_db, events.voice_events = real


def shortlist(job: Path, withhold_clips: bool = False):
    """(Signals, hot spots) for a job from its saved sound, by the engine's own code."""
    from streamclip import signals
    from streamclip.schema import ViewerClip
    job = Path(job)
    stream = _read(job / "stream.json")
    wanted = int((_read(job / "request.json") or {}).get("clips") or 6)
    chat = _read(job / "source" / "chat.json")
    clips = None if withhold_clips else _read(job / "source" / "viewer_clips.json")
    with saved_sound(job):
        curve = signals.build_signals(job / "source" / "audio.m4a", float(stream["duration"]), chat,
                                      [ViewerClip.from_dict(c) for c in clips] if clips else None)
    return curve, signals.pick_hotspots(curve, wanted)


def rescored(job: Path) -> dict:
    row = measure(job, [h.to_dict() for h in shortlist(job)[1]])
    row["kept"] = row["keep_rate"] = None       # the judge's answer was for the old shortlist
    clips = _read(Path(job) / "source" / "viewer_clips.json")
    if clips:
        blind = [(h.start, h.end) for h in shortlist(job, withhold_clips=True)[1]]
        row["blind_hit_views"] = hits(clips, blind)[0]
        row["blind_hit_rate"] = row["blind_hit_views"] / row["clip_views"]
    return row


def refind(job: Path, progress=print) -> None:
    """signals.json, hotspots.json and the candidates again, from the sound already fetched."""
    from clipper.jobs import write_json
    from streamclip import windows
    from streamclip.jobdir import JobDir
    from streamclip.source import open_source
    folder = JobDir(Path(job))
    curve, hotspots = shortlist(folder.root)
    write_json(folder.signals, curve.to_dict())
    wanted = int((_read(folder.request) or {}).get("clips") or 6)
    write_json(folder.hotspots, {"wanted": wanted, "hotspots": [h.to_dict() for h in hotspots]})
    for stale in (folder.picks, folder.judge_raw, folder.camera):
        stale.unlink(missing_ok=True)
    windows.build_candidates(folder, open_source(_read(folder.request)["source"]), hotspots,
                             progress=progress)
    folder.set_status("judge", "ready")


# --- a timed run ---------------------------------------------------------------------------

def timed_run(source: str, job: Path, wanted: int = 6, only: tuple[str, ...] = STAGES) -> int:
    """find, judge and make, with each stage's and each step's seconds written to
    <job>/tuning.json, and the sound's scores kept for --rescore."""
    from streamclip import stages
    from streamclip.jobdir import JobDir
    from streamclip.signals import events, loudness
    from streamclip.source import extras

    folder = JobDir(Path(job))
    record = _read(folder.root / TIMINGS) or {}
    record.setdefault("stages", {})
    record.setdefault("steps", {})
    began = time.monotonic()
    mark = {"key": None, "at": began}
    steps: dict[str, float] = {}

    def moved(stage: str, step: str, detail: str = "") -> None:
        now = time.monotonic()
        if mark["key"]:
            steps[mark["key"]] = steps.get(mark["key"], 0.0) + now - mark["at"]
        what = detail if step.startswith("clip ") else step     # make: by what is being done
        mark["key"], mark["at"] = (f"{stage}/{what}" if what else None), now

    real_status = JobDir.set_status

    def set_status(self, stage, step, detail="", error=None):
        if self.root == folder.root:
            moved(stage, step, detail)
        return real_status(self, stage, step, detail, error)

    real_chat = extras.download_chat

    def download_chat(*args, **kwargs):
        start = time.monotonic()
        try:
            return real_chat(*args, **kwargs)
        finally:
            record["chat_seconds"] = round(time.monotonic() - start, 1)

    kept: dict[str, object] = {}
    real_db, real_voice = loudness.per_second_db, events.voice_events

    def per_second_db(*args, **kwargs):
        kept["db"] = real_db(*args, **kwargs)
        return kept["db"]

    def voice_events(*args, **kwargs):
        kept["track"] = real_voice(*args, **kwargs)
        return kept["track"]

    def say(line: str) -> None:
        print(f"[{time.monotonic() - began:7.1f}] {line}", flush=True)

    JobDir.set_status, extras.download_chat = set_status, download_chat
    loudness.per_second_db, events.voice_events = per_second_db, voice_events
    code = 0
    try:
        calls = {"find": lambda: stages.find(source, folder, wanted=wanted, progress=say),
                 "judge": lambda: stages.judge(folder, progress=say),
                 "make": lambda: stages.make(folder, progress=say)}
        for name in only:
            start = time.monotonic()
            try:
                calls[name]()
            except Exception as exc:        # noqa: BLE001 - the times so far are still written
                say(f"{name} failed: {exc}")
                code = 1
            moved("", "")
            record["stages"][name] = round(time.monotonic() - start, 1)
            record["steps"] = {**{k: v for k, v in record["steps"].items()
                                  if not k.startswith(f"{name}/")},
                               **{k: round(v, 1) for k, v in steps.items() if k.startswith(f"{name}/")}}
            if "db" in kept and "track" in kept:
                _save_scores(folder.root / CACHE, kept["db"], kept["track"])
            (folder.root / TIMINGS).write_text(json.dumps(record, indent=2), encoding="utf-8")
            if code:
                break
    finally:
        JobDir.set_status, extras.download_chat = real_status, real_chat
        loudness.per_second_db, events.voice_events = real_db, real_voice
    return code


# --- the command line ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from clipper import shell
    argv = list(sys.argv[1:] if argv is None else argv)
    shell.tools_on_path()               # the downloader runs ffmpeg by name
    if argv[:1] == ["run"]:
        parser = argparse.ArgumentParser(prog="stream_tune.py run")
        parser.add_argument("source")
        parser.add_argument("--job", required=True)
        parser.add_argument("--clips", type=int, default=6)
        parser.add_argument("--only", nargs="+", choices=STAGES, default=list(STAGES))
        args = parser.parse_args(argv[1:])
        return timed_run(args.source, Path(args.job).expanduser(), args.clips, tuple(args.only))
    if argv[:1] == ["refind"]:
        parser = argparse.ArgumentParser(prog="stream_tune.py refind")
        parser.add_argument("jobs", nargs="+")
        for job in parser.parse_args(argv[1:]).jobs:
            refind(Path(job).expanduser(), progress=lambda line: print(line, flush=True))
        return 0
    parser = argparse.ArgumentParser(prog="stream_tune.py", description=__doc__.split("\n")[0])
    parser.add_argument("jobs", help="the folder that holds the job folders")
    parser.add_argument("--rescore", action="store_true",
                        help="build each shortlist again from the saved sound")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                        help="constants to try, such as curve.WEIGHTS.voice=0.4")
    args = parser.parse_args(argv)
    apply_settings(args.set)
    jobs = jobs_under(Path(args.jobs).expanduser())
    rescore = args.rescore or bool(args.set)
    rows = [rescored(job) if rescore else measure(job) for job in jobs]
    print(table(rows + [total(rows)], blind=rescore))
    return 0


if __name__ == "__main__":
    sys.exit(main())
