"""One worker thread and a queue: the test page runs one job (or one remake) at a time.

The stages write their own status as they go. The worker adds what they cannot: `queued`
before a job starts, `failed` with the reason when a stage raises, and `done` when a stage
returns without saying so.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

from clipper import shell
from clipper.jobs import write_json
from streamclip import stages
from streamclip.jobdir import JobDir
from streamclip.schema import MakeOptions

RUNNING = ("find", "judge", "make")
LOG_LINES = 200


@dataclass(frozen=True)
class _Run:
    job: JobDir
    source: str
    wanted: int
    creator: str


@dataclass(frozen=True)
class _Remake:
    job: JobDir
    rank: int
    options: MakeOptions


def mark_queued(job: JobDir, step: str = "", detail: str = "") -> None:
    """`queued` is the page's own stage, so it is written here and not through set_status."""
    write_json(job.status, {"stage": "queued", "step": step, "detail": detail, "error": None,
                            "updated": datetime.now(UTC).isoformat()})


def mark_failed(job: JobDir, reason: str) -> None:
    """Failed, keeping which stage and step it was in, as the command line does."""
    before = job.get_status()
    if before.get("stage") in RUNNING:
        job.set_status("failed", before["stage"], before.get("step", ""), error=reason)
    else:
        job.set_status("failed", "", error=reason)


def reason_for(exc: BaseException) -> str:
    if isinstance(exc, shell.CommandError):
        return shell.describe_failure(exc)
    return str(exc) or type(exc).__name__


class Runner:
    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pending = 0
        self._thread: threading.Thread | None = None
        self._logs: dict[str, deque[str]] = {}

    def run(self, job: JobDir, source: str, wanted: int, creator: str) -> None:
        mark_queued(job)
        self._put(_Run(job, source, wanted, creator))

    def remake(self, job: JobDir, rank: int, options: MakeOptions) -> None:
        mark_queued(job, "make", f"clip {rank}")
        self._put(_Remake(job, rank, options))

    def log(self, job: JobDir) -> list[str]:
        with self._lock:
            return list(self._logs.get(job.root.name, ()))

    def wait_idle(self, timeout: float | None = None) -> bool:
        """True once nothing is queued or running; False if `timeout` ran out first."""
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout)

    def _put(self, task) -> None:
        with self._lock:
            self._pending += 1
            self._logs[task.job.root.name] = deque(maxlen=LOG_LINES)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="streamclip-web", daemon=True)
                self._thread.start()
        self._queue.put(task)

    def _work(self) -> None:
        while True:
            task = self._queue.get()
            try:
                self._do(task)
            finally:
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def _do(self, task) -> None:
        job = task.job
        lines = self._logs.get(job.root.name)

        def progress(line: str) -> None:
            if lines is not None:
                lines.append(str(line))

        try:
            if isinstance(task, _Run):
                stages.run(task.source, job, wanted=task.wanted, creator=task.creator,
                           progress=progress)
            else:
                stages.make(job, only=task.rank, options=task.options, progress=progress)
        except Exception as exc:                     # the worker must outlive any job
            try:
                mark_failed(job, reason_for(exc))
            except Exception:                        # the folder may be gone; nothing to show
                pass
            return
        if job.get_status().get("stage") != "done":
            job.set_status("done", "make" if isinstance(task, _Remake) else "")
