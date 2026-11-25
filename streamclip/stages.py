"""The three stages and the commands that run them.

Each stage reads the job folder's earlier files and writes its own, so any stage can be
re-run without redoing the ones before it. `progress` takes one short line at a time.

A stage never goes quiet for long: every step sets the job's status and says a line, and a
step that runs on without one gets a "Still working" line every `QUIET_AFTER` seconds. A
machine that cuts off a silent job, and a person following `streamclip status`, both need that.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clipper import shell
from clipper.jobs import read_json, write_json
from streamclip.errors import NoVideo, Refused, StageError
from streamclip.jobdir import JobDir, nn, slug
from streamclip.schema import (
    CameraBox, CandidateIndex, ClipResult, Hotspot, MakeOptions, Pick, StreamInfo, Transcript,
)
from streamclip.source import open_source

QUIET_AFTER = 20.0              # seconds without a line before a stage says it is still working
EXTRAS = ("chat", "viewer_clips", "avatar")
NOTHING_STOOD_OUT = "Nothing in this stream stood out enough to clip."
NO_MODEL = "none"               # picks.json's "model" when the picks were made without one
NO_KEY = "No model key. Ranking by excitement only."
NO_CAPTIONS = "This stream is not in Latin letters, so the clips have no captions."
CLIP_HEIGHT = 1080              # the tallest a clip's source stretch is fetched at
PCM_RATE = 16000                # the clip's own sound, as the text and cut code measure it
MOTION_FPS = 10.0               # how often the picture's movement is sampled for dead air


def _why(exc: BaseException) -> str:
    """One plain line for what went wrong."""
    if isinstance(exc, shell.CommandError):
        return shell.describe_failure(exc)
    return str(exc) or type(exc).__name__


class _Report:
    """A stage's voice: its status file, its progress lines, and a pulse when it is busy."""

    def __init__(self, job: JobDir | None, stage: str, progress) -> None:
        self.job, self.stage, self._progress = job, stage, progress
        self._step, self._detail = "", ""
        self._lock = threading.Lock()
        self._started = self._spoke = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Report:
        self._thread = threading.Thread(target=self._pulse, name=f"streamclip-{self.stage}-pulse",
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        with self._lock:                # waits out a pulse that is mid-line
            self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def step(self, step: str, line: str | None = None, detail: str = "") -> None:
        """Move to a new step: written to the status file, and said when there is a line."""
        self.status(self.stage, step, detail)
        if line:
            self.say(line)

    def detail(self, detail: str) -> None:
        self.status(self.stage, self._step, detail)

    def status(self, stage: str, step: str, detail: str = "") -> None:
        with self._lock:
            self.stage, self._step, self._detail = stage, step, detail
            if self.job is not None:        # `warm` has no job to write to
                self.job.set_status(stage, step, detail)

    def say(self, line: str) -> None:
        with self._lock:
            self._spoke = time.monotonic()
            self._progress(line)

    def _pulse(self) -> None:
        """Every `QUIET_AFTER` seconds without a line, say one and touch the status file.

        Nothing stops it but the stage ending: a line that could not be said, or a status
        that could not be written, is tried again on the next beat."""
        while not self._stop.wait(min(1.0, QUIET_AFTER / 4)):
            try:
                with self._lock:
                    if time.monotonic() - self._spoke < QUIET_AFTER or self._stop.is_set():
                        continue
                    self._spoke = time.monotonic()
                    where = " · ".join(part for part in (self._step, self._detail) if part)
                    if self.job is not None:
                        self.job.set_status(self.stage, self._step, self._detail)   # a fresh time
                    # Said inside the lock, so it can never come after the stage's last line.
                    self._progress(f"Still working: {where} "
                                   f"({time.monotonic() - self._started:.0f} s so far)")
            except Exception:           # noqa: BLE001 - the pulse outlives a failed write
                pass

def _as_given(source: str) -> str:
    """The source as it is kept in request.json: a link as it came, a file by its full path,
    so a later stage finds it whatever folder it is run from."""
    text = str(source).strip()
    if "://" in text or text.lower().startswith(("twitch.tv/", "www.twitch.tv/", "m.twitch.tv/",
                                                 "clips.twitch.tv/")):
        return text
    return os.path.abspath(os.path.expanduser(text))


def _extra(call) -> tuple[object, str]:
    """One optional fetch. Whatever it does, the job goes on."""
    try:
        value, note = call()
    except Exception as exc:            # noqa: BLE001 - an extra never fails a job
        return None, f"missing: {_why(exc)}"
    return (value, note) if value is not None else (None, note or "missing: nothing came back")


def find(source: str, job: JobDir, wanted: int = 6, creator: str = "", progress=print) -> None:
    """Describe the stream, fetch its sound, build the curve, shortlist and prepare candidates."""
    from streamclip import signals, windows     # here: they load numpy and the model code

    with _Report(job, "find", progress) as report:
        report.step("describe", "Reading the stream…")
        source = _as_given(source)
        write_json(job.request, {"source": source, "clips": int(wanted), "creator": creator})
        src = open_source(source)
        info = src.describe()
        if not info.creator_name:
            info.creator_name = creator
        write_json(job.stream, info.to_dict())
        for stale in (job.picks, job.judge_raw, job.camera):    # they belonged to an earlier find
            stale.unlink(missing_ok=True)
        named = f" by {info.creator_name}" if info.creator_name else ""
        report.say(f"Stream: “{info.title}”{named}, {info.duration / 60:.0f} min.")

        report.step("sound", "Fetching the stream's sound…")
        job.audio.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="find-fetch") as pool:
            sound = pool.submit(src.fetch_audio, job.audio)
            extras = {"chat": pool.submit(_extra, src.chat_times),
                      "viewer_clips": pool.submit(_extra, src.viewer_clips),
                      "avatar": pool.submit(_extra, lambda: src.avatar(job.avatar))}
            try:
                sound.result()
            except Refused:
                raise
            except Exception as exc:
                raise StageError(f"The stream's sound could not be fetched: {_why(exc)}") from exc
            report.say("Sound: fetched.")
            got = {name: fut.result() for name, fut in extras.items()}
        chat, viewer_clips = got["chat"][0], got["viewer_clips"][0]
        if chat is not None:
            write_json(job.chat, [float(t) for t in chat])
        else:
            job.chat.unlink(missing_ok=True)
        if viewer_clips is not None:
            write_json(job.viewer_clips, [clip.to_dict() for clip in viewer_clips])
        else:
            job.viewer_clips.unlink(missing_ok=True)
        if got["avatar"][0] is None:
            job.avatar.unlink(missing_ok=True)
        for name in EXTRAS:
            info.extras[name] = got[name][1]
        write_json(job.stream, info.to_dict())
        report.say("Extras: " + ", ".join(f"{name.replace('_', ' ')} {info.extras[name]}"
                                          for name in EXTRAS) + ".")

        report.step("signals", "Scoring every second of the sound…")
        curve = signals.build_signals(job.audio, info.duration, chat, viewer_clips)
        write_json(job.signals, curve.to_dict())

        report.step("hotspots")
        hotspots = signals.pick_hotspots(curve, wanted)     # the same object: it holds the reasons
        write_json(job.hotspots, {"wanted": int(wanted), "hotspots": [h.to_dict() for h in hotspots]})
        if not hotspots:
            raise StageError(NOTHING_STOOD_OUT)
        report.say(f"Hot spots: {len(hotspots)}.")

        report.step("windows", f"Preparing {len(hotspots)} hot spots for the judge…")

        def window_line(line: str) -> None:
            if str(line).startswith("Windows:"):
                report.detail(str(line).removeprefix("Windows:").strip())
            report.say(line)
        index = windows.build_candidates(job, src, hotspots, progress=window_line)

        report.status("judge", "ready")
        report.say(f"Ready for the judge: {len(index.ready)} of {len(hotspots)} hot spots.")


def judge(job: JobDir, raw: Path | None = None, progress=print) -> None:
    """Turn the model's answer into picks.json. `raw` is an answer already fetched elsewhere.

    A job always leaves here with picks. No key, a call that failed, an answer that could not
    be read or one that left nothing usable: the hot spots are ranked by excitement instead,
    every pick is marked weak, and the model is recorded as "none"."""
    from streamclip.judge import call, checks

    with _Report(job, "judge", progress) as report:
        report.step("reading")
        try:
            wanted = int(read_json(job.hotspots)["wanted"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StageError("This job has no hot spots to judge. Run find first.") from exc

        answer: dict | None = None
        model = NO_MODEL
        if raw is not None:
            report.step("answer", "Reading the judge's answer…")
            try:
                answer = read_json(Path(raw))
                if not isinstance(answer, dict):
                    raise ValueError("not a JSON object")
            except (OSError, ValueError) as exc:
                answer = None
                report.say(f"The judge's answer could not be read ({_why(exc)}). "
                           "Ranking by excitement only.")
        else:
            key = next((os.environ[name].strip() for name in call.KEY_NAMES
                        if os.environ.get(name, "").strip()), "")
            if not key:
                report.say(NO_KEY)
            else:
                report.step("asking", "Asking the judge…")
                try:
                    answer = call.judge(job.root, key, progress=report.say)
                    if not isinstance(answer, dict):
                        raise call.JudgeError("the answer was not the JSON asked for")
                except Exception as exc:        # noqa: BLE001 - a judge that fails never fails the job
                    answer = None
                    report.say(f"The judge gave no usable answer ({_why(exc)}). "
                               "Ranking by excitement only.")

        picks = []
        if answer is not None:
            report.step("checking")
            picks = checks.apply_checks(answer, job, wanted)
            if picks:
                model = str(answer.get("model") or call.MODEL)
            else:
                report.say("The judge kept nothing usable. Ranking by excitement only.")
        # What the judge said is kept even when its picks are not used: it is the record of
        # what was ruled out.
        if answer is not None:
            write_json(job.judge_raw, answer)
        else:
            job.judge_raw.unlink(missing_ok=True)
        if not picks:
            report.step("fallback")
            barred = checks.ruled_out(answer) if answer is not None else set()
            picks = checks.fallback_picks(job, wanted, skip=barred)
            if not picks and barred:
                job.picks.unlink(missing_ok=True)
                raise StageError("The judge ruled out every moment in this stream as not fit to post.")
        if not picks:
            raise StageError("None of this stream's hot spots could be made into a clip.")

        write_json(job.picks, {"model": model, "wanted": wanted,
                               "picks": [pick.to_dict() for pick in picks]})
        weak = sum(1 for pick in picks if pick.weak)
        report.status("make", "ready")
        report.say(f"Picks: {len(picks)}" + (f", {weak} of them weak." if weak else "."))


def make(job: JobDir, out: Path | None = None, only: int | None = None,
         options: MakeOptions = MakeOptions(), progress=print) -> None:
    """Render the picks into finished clips. `only` re-renders one rank.

    One clip failing never stops the others: it becomes a note in clips.json. clips.json
    lists every clip that exists, so re-rendering one keeps the rest listed."""
    from streamclip import sounds

    if out is not None:
        job = JobDir(job.root, out)
    with _Report(job, "make", progress) as report:
        report.step("reading", "Getting ready to make the clips…")
        try:
            request = read_json(job.request)
            info = StreamInfo.from_dict(read_json(job.stream))
            index = CandidateIndex.from_dict(read_json(job.candidates_index))
            hotspots = {h.n: h for h in map(Hotspot.from_dict, read_json(job.hotspots)["hotspots"])}
            picks = [Pick.from_dict(d) for d in read_json(job.picks)["picks"]]
            source = str(request["source"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StageError("This job has no picks to make clips from. "
                             "Run find and judge first.") from exc
        if not picks:
            raise StageError("This job has no picks to make clips from. Run find and judge first.")
        todo = picks if only is None else [p for p in picks if p.rank == only]
        if not todo:
            raise StageError(f"There is no clip {only} in this job. "
                             f"It has clips 1 to {max(p.rank for p in picks)}.")

        src = open_source(source)
        camera = _camera(job, index, report)
        pack = sounds.load_pack() if options.sounds else {}
        avatar = job.avatar if job.avatar.exists() else None

        made: dict[int, ClipResult] = {}
        skipped: dict[int, str] = {}
        said: dict[int, list[str]] = {}
        for pick in todo:
            where = f"clip {pick.rank} of {len(picks)}"
            try:
                said[pick.rank], made[pick.rank] = _make_clip(job, src, source, info, index, hotspots, pick, camera,
                                             pack, avatar, options, report, where)
            except Exception as exc:        # noqa: BLE001 - one clip never stops the rest
                skipped[pick.rank] = _why(exc)
                report.say(f"Clip {nn(pick.rank)} was skipped: {skipped[pick.rank]}")

        report.step("listing")
        listed = _write_listing(job, source, index, picks, made, skipped, said, whole=only is None)
        if not made:
            raise StageError(f"No clip could be made: {next(iter(skipped.values()))}")
        report.status("done", "", f"{listed} clip{'' if listed == 1 else 's'}")
        report.say(f"Done: {listed} clip{'' if listed == 1 else 's'} in {job.out}.")


def _camera(job: JobDir, index: CandidateIndex, report: _Report) -> CameraBox | None:
    """Where the streamer's camera sits, worked out once per job and kept in camera.json."""
    from streamclip import framing

    if job.camera.exists():
        try:
            saved = read_json(job.camera)["camera"]
            return None if saved is None else CameraBox.from_dict(saved)
        except (OSError, ValueError, KeyError, TypeError):
            pass                        # an unreadable file is worked out again
    report.step("camera", "Looking for the streamer's camera…")
    windows = [w for w in (job.candidate(n) / "window.mp4" for n in index.ready) if w.exists()]
    try:
        camera = framing.find_camera(windows)
    except Exception as exc:            # noqa: BLE001 - each clip can still be framed on its own
        report.say(f"The camera could not be looked for ({_why(exc)}). "
                   "Each clip is framed on its own.")
        return None
    write_json(job.camera, {"camera": None if camera is None else camera.to_dict()})
    report.say("Camera: found." if camera is not None else "Camera: none that stays in one place.")
    return camera


def _real_title(title: str, info: StreamInfo) -> str:
    """The pick's title when it is one of its own, else "": no title sticker is drawn and the
    file is named NN-clip.mp4. A pick made without the judge has none, and the stream's own
    title (for a file, the file's name), which stood in for one before, is not one either."""
    from streamclip.judge.checks import _stream_title

    shown = " ".join(str(title or "").split())
    stand_ins = {" ".join(info.title.split()).casefold(), _stream_title(info.to_dict()).casefold()}
    return "" if shown.casefold() in stand_ins else shown


def _make_clip(job: JobDir, src, source: str, info: StreamInfo, index: CandidateIndex,
               hotspots: dict[int, Hotspot], pick: Pick, camera: CameraBox | None, pack: dict,
               avatar: Path | None, options: MakeOptions, report: _Report,
               where: str) -> tuple[list[str], ClipResult]:
    """One pick taken all the way to a finished file. Every step's output stays in clips/NN/."""
    from dataclasses import replace

    from clipper.pacing import TimeMap
    from streamclip import edit, framing, text, windows
    from streamclip.audio import read_pcm
    from streamclip.source import media

    def doing(what: str) -> None:
        report.step(where, f"{where.capitalize()}: {what}", detail=what)

    spot = hotspots.get(pick.candidate)
    if spot is None:
        raise StageError(f"its hot spot ({pick.candidate}) is not in this job any more")
    end = min(pick.end, info.duration) if info.duration > 0 else pick.end
    pick = replace(pick, start=max(0.0, pick.start), end=end,       # never outside the stream
                   title=_real_title(pick.title, info))
    if pick.end <= pick.start:
        raise StageError("its stretch is empty")
    folder = job.clip(pick.rank)
    folder.mkdir(parents=True, exist_ok=True)

    # 1. The stretch of video. Kept between runs, and fetched again when the pick has moved.
    video, stamp = folder / "source.mp4", folder / "source.json"
    asked = {"source": source, "start": round(pick.start, 3), "end": round(pick.end, 3)}
    try:
        have = read_json(stamp) if video.exists() else None
    except (OSError, ValueError):
        have = None
    if have != asked:
        doing("fetching the video")
        stamp.unlink(missing_ok=True)
        video.unlink(missing_ok=True)
        src.fetch_video(pick.start, pick.end, CLIP_HEIGHT, video)
        write_json(stamp, asked)
    probed = media.probe(video)
    pictures = media.streams_of(probed, "video")
    if not pictures:
        raise NoVideo(media.NO_VIDEO)
    fps = media.frame_rate(pictures[0]) or 30.0
    length = float(probed.get("format", {}).get("duration") or 0.0) or pick.end - pick.start
    clip_end = pick.start + min(pick.end - pick.start, length)

    # 2. The layout: the shot cuts from the candidate's small window, the faces from the
    # stretch just fetched, where a corner camera's face is big enough for the face model.
    doing("framing")
    layout = framing.plan_layout(job.candidate(pick.candidate) / "window.mp4", spot.start, pick, camera,
                                 clip_video=video)
    write_json(folder / "layout.json", layout.to_dict())

    # 3 and 4. What is said, how loud it is, how much the picture moves, and so what is cut.
    try:
        transcript = Transcript.from_dict(read_json(job.candidate(pick.candidate) / "transcript.json"))
        words = [w for w in transcript.words if w.end > pick.start and w.start < clip_end]
    except (OSError, ValueError, KeyError, TypeError):
        words = []                      # a clip with no words is still a clip
    # The window was heard fast, for the judge. The clip is heard again, accurately, and
    # those words drive the cut, the captions and the word sizes. The pick does not move.
    doing("listening again for the exact words")
    notes: list[str] = []
    began = time.monotonic()
    try:
        heard = windows.transcribe_clip(job.audio, pick.start, clip_end, info.duration,
                                        language=index.language or None)
        failed = "" if heard else "it heard no words"
    except Exception as exc:            # noqa: BLE001 - the first listen's words still make a clip
        heard, failed = [], _why(exc)
    if heard:
        words = heard
    elif words:
        notes.append(f"Clip {nn(pick.rank)} uses the first listen's words: "
                     f"the second listen did not work ({failed}).")
    report.say(f"Clip {nn(pick.rank)}: second listen, {len(heard)} words in "
               f"{time.monotonic() - began:.1f} s.")
    write_json(folder / "words.json", {"pass": "second" if heard else "first",
                                       "words": [w.to_dict() for w in words]})
    doing("cutting dead air")
    pcm = read_pcm(video, rate=PCM_RATE)
    motion = edit.motion_series(video, fps=MOTION_FPS)
    keep = edit.keep_intervals(words, pick.start, clip_end, pcm, PCM_RATE, motion, MOTION_FPS,
                               pick.protect, fps, options.dead_air)
    if not keep:
        raise StageError("nothing was left of it after the dead air was cut")
    write_json(folder / "cut.json", {"keep": [[a, b] for a, b in keep], "fps": fps})
    time_map = TimeMap(keep)

    # 5 and 6. Text, sounds and the zoom.
    doing("drawing the text")
    for old in ("captions.ass", "title.png", "tag.png"):
        (folder / old).unlink(missing_ok=True)
    assets = text.build_text(folder, pick, words, pcm, PCM_RATE, time_map, info, avatar,
                             index.captions, options.speed)
    placed = edit.place_sounds(pick.beats, pick.start, time_map, options.speed, pack) if pack else []
    zoom_t = edit.zoom_at(pick.beats, pick.start, pcm, PCM_RATE)

    # 7. The encode, then the caption beside it.
    doing("rendering")
    name = f"{nn(pick.rank)}-{slug(pick.title)}"
    dest = job.out / f"{name}.mp4"
    edit.render_clip(video, layout, keep, zoom_t, assets, placed, options.speed, dest)
    caption = dest.with_suffix(".txt")
    lines = [pick.caption.strip() or pick.title.strip()]
    if pick.hashtags:
        lines += ["", " ".join(pick.hashtags)]
    caption.write_text("\n".join(lines) + "\n", encoding="utf-8")

    finished = float(shell.probe(dest)["format"]["duration"])
    kept = sum(b - a for a, b in keep)
    gaps = sum(1 for (_, b), (a, _) in zip(keep, keep[1:]) if a > b)
    gaps += (keep[0][0] > 1e-6) + (clip_end - pick.start - keep[-1][1] > 1.0 / fps)
    kinds = list(dict.fromkeys(shot.kind for shot in layout.shots))
    named = f": “{pick.title}”," if pick.title else ","
    report.say(f"Clip {nn(pick.rank)} done{named} {finished:.1f} s.")
    return notes, ClipResult(
        rank=pick.rank, file=dest.name, caption_file=caption.name, title=pick.title,
        score=pick.score, reason=pick.reason, source_start=pick.start, source_end=pick.end,
        duration=round(finished, 2), weak=pick.weak, layout=kinds, cuts=int(gaps),
        seconds_cut=round(max(0.0, clip_end - pick.start - kept), 2),
        sounds=[p.sound for p in placed], captions=assets.ass is not None)


def _write_listing(job: JobDir, source: str, index: CandidateIndex, picks: list[Pick],
                   made: dict[int, ClipResult], skipped: dict[int, str],
                   said: dict[int, list[str]], whole: bool) -> int:
    """out/clips.json with every clip that exists. Gives how many are listed.

    A full run lists what it made. Re-making one clip keeps the others as they were listed,
    and a re-make that failed keeps the clip that was there. Files an earlier listing of this
    same source named and this one does not are removed, so `out/` only holds the clips the
    list describes. A list left there by another source is never read or acted on."""
    listing = job.out / "clips.json"
    try:
        before = read_json(listing)
        if before.get("source") != source:
            raise ValueError("the list is another job's")
        old_clips = [ClipResult.from_dict(c) for c in before["clips"]]
        old_notes = [str(n) for n in before.get("notes", [])]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        old_clips, old_notes = [], []

    ranks = {p.rank for p in picks}
    clips = dict(made)
    notes = {rank: [f"Clip {nn(rank)} was skipped: {why}"] for rank, why in skipped.items()}
    notes.update({rank: list(lines) for rank, lines in said.items() if rank in made and lines})
    if not whole:
        for clip in old_clips:
            if clip.rank in ranks and clip.rank not in made and (job.out / clip.file).exists():
                clips[clip.rank] = clip
        for rank in ranks - set(made) - set(skipped):
            notes[rank] = [n for n in old_notes if n.startswith(f"Clip {nn(rank)} ")
                           and not (rank in clips and " was skipped: " in n)]

    named = {name for clip in clips.values() for name in (clip.file, clip.caption_file)}
    for clip in old_clips:
        for name in (clip.file, clip.caption_file):
            # Only a plain file name this stage could have written is ever removed.
            if name not in named and name == Path(name).name and name[:2].isdigit():
                (job.out / name).unlink(missing_ok=True)

    ordered = [clips[rank] for rank in sorted(clips)]
    write_json(listing, {
        "source": source,
        "clips": [clip.to_dict() for clip in ordered],
        "captions": bool(index.captions),
        "notes": ([] if index.captions else [NO_CAPTIONS])
                 + [note for rank in sorted(notes) for note in notes[rank]],
    })
    return len(ordered)


def run(source: str, job: JobDir, out: Path | None = None, wanted: int = 6, creator: str = "",
        options: MakeOptions = MakeOptions(), progress=print) -> None:
    """find, judge and make in one go."""
    find(source, job, wanted=wanted, creator=creator, progress=progress)
    judge(job, progress=progress)
    make(job, out=out, options=options, progress=progress)


def warm(progress=print) -> None:
    """Download every model once, so no job has to."""
    from clipper import provision
    from streamclip import signals, windows

    with _Report(None, "warm", progress) as report:
        report.step("the sound classifier", "Sound classifier…")
        signals.ensure_model()
        report.step("the speech models", "Speech models…")
        windows.ensure_model()
        report.step("the face model", "Face model…")
        provision.ensure(("model",))
        if not _ffmpeg_has_subtitles():
            report.step("ffmpeg", "An ffmpeg that can burn in captions…")
            provision.ensure(("ffmpeg",))
    progress("Models are ready.")


def _ffmpeg_has_subtitles() -> bool:
    """Whether the ffmpeg in use can burn in captions (built with libass). Many system builds,
    Homebrew's slimmer ones among them, can't, and every clip needs it."""
    try:
        return " subtitles " in shell.run([shell.ffmpeg(), "-hide_banner", "-filters"], timeout=30)
    except Exception:
        return False
