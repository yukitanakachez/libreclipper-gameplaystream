"""The stream clipper's local test page: a small API over job folders, and one plain page.

Every job is a folder under `jobs_root`, named `<slug of the source>-<YYYYmmdd-HHMMSS>`.
Dropped files are kept in `jobs_root/uploads/`. Nothing here reaches outside `jobs_root`.
"""

from __future__ import annotations

import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

from clipper.jobs import read_json, write_json
from streamclip import settings
from streamclip.jobdir import JobDir, nn, slug
from streamclip.schema import MakeOptions
from streamclip.web.runner import RUNNING, Runner, mark_failed

PAGE = Path(__file__).with_name("index.html")
UPLOADS = "uploads"
CURVE_POINTS = 1200
_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
_STOPPED = "The test page was stopped while this job was running. Run it again."


class JobRequest(BaseModel):
    source: str
    clips: int = Field(settings.DEFAULT_CLIPS, ge=settings.MIN_CLIPS, le=settings.MAX_CLIPS)
    creator: str = ""

    @field_validator("source", "creator")
    @classmethod
    def _trimmed(cls, value: str) -> str:
        return value.strip()

    @field_validator("source")
    @classmethod
    def _given(cls, value: str) -> str:
        if not value:
            raise ValueError("Paste a Twitch link or drop a video file first.")
        return value


class Knobs(BaseModel):
    speed: float = Field(settings.SPEED, ge=0.5, le=2.0)
    dead_air: float = Field(settings.DEAD_AIR, ge=0.1, le=5.0)
    sounds: bool = True


def _read(path: Path):
    """A job file's contents, or None while it does not exist (or is half written)."""
    try:
        return read_json(path)
    except (FileNotFoundError, ValueError):
        return None


def _new_id(jobs_root: Path, source: str) -> str:
    """Make the job's folder and return its name. Two jobs in one second get -2, -3 ..."""
    base = f"{slug(source)[:40].rstrip('-') or 'clip'}-{datetime.now():%Y%m%d-%H%M%S}"
    job_id, k = base, 2
    while True:
        try:
            (jobs_root / job_id).mkdir(parents=True)
            return job_id
        except FileExistsError:
            job_id, k = f"{base}-{k}", k + 1


def _upload_name(name: str) -> tuple[str, str]:
    base = Path(name.replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._")[:80] or "video"
    suffix = re.sub(r"[^A-Za-z0-9.]+", "", Path(base).suffix)[:10] or ".mp4"
    return stem, suffix


def downsample(values: list[float], most: int = CURVE_POINTS) -> list[float]:
    """At most `most` points; each bucket keeps its highest value, so no peak disappears."""
    if len(values) <= most:
        return [round(float(v), 4) for v in values]
    size = math.ceil(len(values) / most)
    return [round(float(max(values[i:i + size])), 4) for i in range(0, len(values), size)]


def judged_rows(job: JobDir) -> list[dict] | None:
    """One row per candidate: why it was shortlisted, what the judge said, where it ranked."""
    hotspots = _read(job.hotspots)
    if hotspots is None:
        return None
    raw = _read(job.judge_raw) or {}
    answers = {}
    for c in raw.get("candidates") or []:
        if isinstance(c, dict) and isinstance(c.get("n"), int):
            answers.setdefault(c["n"], c)
    picks = {}
    for p in (_read(job.picks) or {}).get("picks") or []:
        picks.setdefault(p.get("candidate"), p)
    rows = []
    for h in hotspots.get("hotspots") or []:
        n = h["n"]
        answer, pick = answers.get(n, {}), picks.get(n)
        sheet = job.candidate(n) / "sheet.jpg"
        rows.append({
            "n": n,
            "peak": h.get("peak"),
            "score_signal": h.get("score"),
            "fired": h.get("fired") or [],
            "keep": answer.get("keep"),
            "score": answer.get("score"),
            "reason": answer.get("reason"),
            "rank": pick.get("rank") if pick else None,
            "weak": pick.get("weak") if pick else None,
            "sheet": f"candidates/{nn(n)}/sheet.jpg" if sheet.is_file() else None,
        })
    # Picked first by rank, then the rest by the judge's score, then by number.
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0,
                             -(r["score"] if isinstance(r["score"], (int, float)) else -1), r["n"]))
    return rows


def create_app(jobs_root: Path) -> FastAPI:
    jobs_root = Path(jobs_root)
    jobs_root.mkdir(parents=True, exist_ok=True)
    root = jobs_root.resolve()
    runner = Runner()
    making_ids = threading.Lock()
    app = FastAPI(title="Stream clipper test page")
    app.state.runner = runner
    app.state.jobs_root = jobs_root

    def job_folders() -> list[Path]:
        return [p for p in jobs_root.iterdir()
                if p.is_dir() and p.name != UPLOADS and _ID.fullmatch(p.name)
                and ((p / "request.json").exists() or (p / "status.json").exists())]

    # A job the last page left queued or running will never finish: say so.
    for folder in job_folders():
        job = JobDir(folder)
        if job.get_status().get("stage") in ("queued", *RUNNING):
            mark_failed(job, _STOPPED)

    def job_for(job_id: str) -> JobDir:
        if not _ID.fullmatch(job_id) or job_id == UPLOADS or not (jobs_root / job_id).is_dir():
            raise HTTPException(404, "There is no such job.")
        return JobDir(jobs_root / job_id)

    @app.get("/", response_class=HTMLResponse)
    def page():
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    @app.post("/api/upload")
    async def upload(request: Request, name: str):
        """Keep a dropped file: a browser hands over its contents but never its place on disk."""
        folder = jobs_root / UPLOADS
        folder.mkdir(parents=True, exist_ok=True)
        stem, suffix = _upload_name(name)
        with making_ids:                             # reserve a name no other upload has
            dest, k = folder / f"{stem}{suffix}", 2
            while True:
                try:
                    os.close(os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                    break
                except FileExistsError:
                    dest, k = folder / f"{stem}-{k}{suffix}", k + 1
        partial = dest.with_name(dest.name + ".partial")
        done = False
        try:
            with open(partial, "wb") as out:
                async for chunk in request.stream():
                    out.write(chunk)
            if partial.stat().st_size == 0:
                raise HTTPException(400, "That file is empty.")
            partial.replace(dest)
            done = True
        finally:
            partial.unlink(missing_ok=True)
            if not done:
                dest.unlink(missing_ok=True)
        return {"path": str(dest.resolve())}

    @app.post("/api/jobs")
    def start_job(body: JobRequest):
        with making_ids:
            job_id = _new_id(jobs_root, body.source)
        job = JobDir(jobs_root / job_id)
        write_json(job.request, {"source": body.source, "clips": body.clips, "creator": body.creator})
        runner.run(job, body.source, body.clips, body.creator)
        return {"id": job_id}

    @app.get("/api/jobs")
    def list_jobs():
        def started(folder: Path) -> int:
            marker = folder / "request.json"
            return (marker if marker.exists() else folder).stat().st_mtime_ns

        listed = []
        for folder in sorted(job_folders(), key=started, reverse=True):
            job = JobDir(folder)
            request = _read(job.request) or {}
            source = request.get("source") or (_read(job.stream) or {}).get("source", "")
            listed.append({"id": folder.name, "source": source, "status": job.get_status()})
        return listed

    @app.get("/api/jobs/{job_id}")
    def job_report(job_id: str):
        job = job_for(job_id)
        return {
            "id": job_id,
            "request": _read(job.request),
            "status": job.get_status(),
            "stream": _read(job.stream),
            "hotspots": _read(job.hotspots),
            "judged": judged_rows(job),
            "picks": _read(job.picks),
            "clips": _read(job.out / "clips.json"),
            "log": runner.log(job),
        }

    @app.get("/api/jobs/{job_id}/curve")
    def curve(job_id: str):
        signals = _read(job_for(job_id).signals)
        if signals is None:
            raise HTTPException(404, "This job has no excitement curve yet.")
        values = signals.get("excitement") or []
        return {"duration": float(signals.get("duration") or len(values)), "points": downsample(values)}

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def job_file(job_id: str, path: str):
        folder = job_for(job_id).root.resolve()
        wanted = (folder / path).resolve()
        if not wanted.is_relative_to(folder) or not wanted.is_relative_to(root) or not wanted.is_file():
            raise HTTPException(404, "There is no such file in this job.")
        return FileResponse(wanted)

    @app.post("/api/jobs/{job_id}/clips/{rank}/remake")
    def remake(job_id: str, rank: int, knobs: Knobs):
        job = job_for(job_id)
        ranks = {p.get("rank") for p in (_read(job.picks) or {}).get("picks") or []}
        if rank not in ranks:
            raise HTTPException(404, f"This job has no clip {rank}.")
        runner.remake(job, rank, MakeOptions(speed=knobs.speed, dead_air=knobs.dead_air,
                                             sounds=knobs.sounds))
        return {"ok": True}

    return app
