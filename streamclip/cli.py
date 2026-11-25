"""Command line entry point: streamclip find / judge / make / run / status / warm / serve.

Exit codes: 0 done, 2 refused (the request cannot be clipped; the reason is on stderr),
1 anything else (the reason is on stderr and in the job's status.json).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from clipper import shell
from streamclip import settings, stages
from streamclip.errors import Refused
from streamclip.jobdir import JobDir, slug
from streamclip.schema import MakeOptions

JOBS = "stream-jobs"                # where `run` and `serve` keep jobs when not told
RUNNING = ("find", "judge", "make")


def _clips(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if not settings.MIN_CLIPS <= n <= settings.MAX_CLIPS:
        raise argparse.ArgumentTypeError(
            f"ask for {settings.MIN_CLIPS} to {settings.MAX_CLIPS} clips, not {text!r}")
    return n


def _above_zero(kind):
    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            value = 0
        if not value > 0:
            raise argparse.ArgumentTypeError(f"must be a number above zero, not {text!r}")
        return value
    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamclip", description="Turn one past Twitch broadcast into finished vertical clips.")
    sub = parser.add_subparsers(dest="command", required=True)

    def asking(p: argparse.ArgumentParser) -> None:
        p.add_argument("--clips", type=_clips, default=settings.DEFAULT_CLIPS, metavar="N",
                       help=f"how many clips ({settings.MIN_CLIPS} to {settings.MAX_CLIPS}, "
                            f"default {settings.DEFAULT_CLIPS})")
        p.add_argument("--creator", default="", metavar="NAME",
                       help="the creator's name for the tag, when the source does not give one")

    def making(p: argparse.ArgumentParser) -> None:
        p.add_argument("--speed", type=_above_zero(float), default=settings.SPEED,
                       help=f"playback speed (default {settings.SPEED})")
        p.add_argument("--dead-air", type=_above_zero(float), default=settings.DEAD_AIR, metavar="SECONDS",
                       help=f"cut dead air longer than this (default {settings.DEAD_AIR})")
        p.add_argument("--no-sounds", action="store_true", help="leave the sound effects out")

    find = sub.add_parser("find", help="shortlist a stream's hot spots and prepare them for the judge")
    find.add_argument("source", help="a link to a past Twitch broadcast, or a video file")
    find.add_argument("--job", required=True, metavar="DIR", help="the job folder to write into")
    asking(find)

    judge = sub.add_parser("judge", help="turn the judge's answer into the chosen moments")
    judge.add_argument("job", help="the job folder")
    judge.add_argument("--raw", metavar="FILE", help="an answer the judge already gave, as a JSON file")

    make = sub.add_parser("make", help="render the chosen moments into finished clips")
    make.add_argument("job", help="the job folder")
    make.add_argument("--out", metavar="DIR", help="where the clips go (default: <job>/out)")
    make.add_argument("--only", type=_above_zero(int), metavar="RANK", help="render just this clip again")
    making(make)

    run = sub.add_parser("run", help="find, judge and make in one go")
    run.add_argument("source", help="a link to a past Twitch broadcast, or a video file")
    run.add_argument("--job", metavar="DIR", help=f"the job folder (default: ./{JOBS}/<name from the source>)")
    run.add_argument("--out", metavar="DIR", help="where the clips go (default: <job>/out)")
    asking(run)
    making(run)

    status = sub.add_parser("status", help="print where a job is, in one line")
    status.add_argument("job", help="the job folder")

    sub.add_parser("warm", help="download the models once, so no job has to")

    serve = sub.add_parser("serve", help="start the local test page")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--jobs", metavar="DIR", default=JOBS, help=f"where jobs are kept (default: ./{JOBS})")
    return parser


def job_folder_for(source: str) -> Path:
    """Where `run` keeps a source's job when --job is not given. The same source gets the same
    folder again, so a stage can be re-run; the digest keeps two sources that read alike apart."""
    link = source.strip().lower().startswith(("http://", "https://"))
    plain = source.strip() if link else str(Path(source).expanduser().resolve())
    name = re.sub(r"^https?://(www\.)?", "", plain, flags=re.I) if link else Path(plain).stem
    digest = hashlib.sha256(plain.encode()).hexdigest()[:8]
    return Path(JOBS) / f"{slug(name)[:40].rstrip('-')}-{digest}"


def _path(text: str | None) -> Path | None:
    return Path(text) if text else None


def _options(args) -> MakeOptions:
    return MakeOptions(speed=args.speed, dead_air=args.dead_air, sounds=not args.no_sounds)


def _progress(line: str) -> None:
    print(line, flush=True)


def _job(args) -> JobDir | None:
    """The job a command works on; None for the commands that have none."""
    if args.command in ("warm", "serve"):
        return None
    if args.command == "run" and not args.job:
        return JobDir(job_folder_for(args.source), _path(args.out))
    return JobDir(Path(args.job), _path(getattr(args, "out", None)))


def _cmd_find(args, job: JobDir) -> int:
    stages.find(args.source, job, wanted=args.clips, creator=args.creator, progress=_progress)
    return 0


def _cmd_judge(args, job: JobDir) -> int:
    stages.judge(job, raw=_path(args.raw), progress=_progress)
    return 0


def _cmd_make(args, job: JobDir) -> int:
    stages.make(job, out=_path(args.out), only=args.only, options=_options(args), progress=_progress)
    return 0


def _cmd_run(args, job: JobDir) -> int:
    stages.run(args.source, job, out=_path(args.out), wanted=args.clips, creator=args.creator,
               options=_options(args), progress=_progress)
    return 0


def _cmd_status(_args, job: JobDir) -> int:
    status = job.get_status()
    if status["stage"] == "failed":
        print(f"failed: {status['error']}")
    else:
        print(" ".join(part for part in (status["stage"], status["step"], status["detail"]) if part))
    return 0


def _cmd_warm(_args, _job) -> int:
    stages.warm(progress=_progress)
    return 0


def _cmd_serve(args, _job) -> int:
    import uvicorn

    from streamclip.web.app import create_app       # only here: the page is not needed to clip

    print(f"The stream clipper's test page is at http://127.0.0.1:{args.port}  (Ctrl-C to stop)",
          flush=True)
    uvicorn.run(create_app(Path(args.jobs)), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


COMMANDS = {
    "find": _cmd_find,
    "judge": _cmd_judge,
    "make": _cmd_make,
    "run": _cmd_run,
    "status": _cmd_status,
    "warm": _cmd_warm,
    "serve": _cmd_serve,
}


def _reason(exc: Exception) -> str:
    if isinstance(exc, shell.CommandError):
        return shell.describe_failure(exc)
    return str(exc) or type(exc).__name__


def _record(job: JobDir | None, reason: str) -> None:
    """Mark the job failed, keeping which stage and step it was in."""
    if job is None:
        return
    try:
        before = job.get_status()
        if before["stage"] in RUNNING:
            job.set_status("failed", before["stage"], before["step"], error=reason)
        else:
            job.set_status("failed", "", error=reason)
    except Exception:               # the reason is already on stderr; never hide it behind this
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    shell.tools_on_path()
    job = _job(args)
    try:
        return COMMANDS[args.command](args, job)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        _record(job, str(exc))
        return 2
    except Exception as exc:                        # a CLI shows the message, not a traceback
        print(f"error: {_reason(exc)}", file=sys.stderr)
        if args.command != "status":
            _record(job, _reason(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
