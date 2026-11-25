"""Every hot spot made into a candidate: `candidates/NN/` with transcript.json, window.mp4
and sheet.jpg, plus `candidates/index.json` saying which are ready.

Video stretches are fetched a few at a time in background threads, since that is waiting on
the network. Transcription runs one window at a time in the calling thread, since the model
already uses every processor. One candidate failing never stops the others.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

from clipper import shell
from clipper.jobs import write_json
from streamclip.errors import StageError
from streamclip.jobdir import JobDir, nn
from streamclip.protocols import Source
from streamclip.schema import CandidateIndex, Hotspot
from streamclip.windows import transcribe
from streamclip.windows.sheet import contact_sheet

FETCHERS = 4                    # video stretches fetched at once
WINDOW_HEIGHT = 360             # the judge only needs to see what happened
FILES = ("transcript.json", "window.mp4", "sheet.jpg")
DEFAULT_LANGUAGE = "en"


def _reason(exc: BaseException) -> str:
    return shell.describe_failure(exc) if isinstance(exc, Exception) else repr(exc)


def build_candidates(job: JobDir, source: Source, hotspots: list[Hotspot], progress=print,
                     model=None) -> CandidateIndex:
    """Transcript, small video and contact sheet for each hot spot; writes the index.

    Raises StageError when not one hot spot could be prepared."""
    total = len(hotspots)
    for h in hotspots:                                  # a re-run starts each folder clean
        for name in FILES:
            (job.candidate(h.n) / name).unlink(missing_ok=True)

    def fetch(h: Hotspot):
        if h.end <= h.start:
            raise ValueError(f"its window is empty ({h.start:.1f}-{h.end:.1f} s)")
        dest = job.candidate(h.n) / "window.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        return source.fetch_video(h.start, h.end, WINDOW_HEIGHT, dest)

    ready: list[int] = []
    notes: list[str] = []
    texts: list[str] = []
    language: str | None = None         # settled by the first window that had any words
    detected: str | None = None         # the first answer at all, for a stream with no speech
    listening = 0.0                     # seconds the first listen took, all windows together

    pool = ThreadPoolExecutor(max_workers=FETCHERS, thread_name_prefix="window-fetch")
    try:
        fetches: list[Future] = [pool.submit(fetch, h) for h in hotspots]
        for done, (h, fut) in enumerate(zip(hotspots, fetches), start=1):
            d = job.candidate(h.n)
            try:
                video = fut.result()
                began = time.monotonic()
                text, heard = _transcribe(job, h, language, model)
                listening += time.monotonic() - began
                detected = detected or heard
                if text and language is None:
                    language = heard
                contact_sheet(video, d / "sheet.jpg")
                if all((d / name).exists() for name in FILES):
                    ready.append(h.n)
                    texts.append(text)
                else:
                    raise StageError("a file is missing after it was made")
            except Exception as exc:        # noqa: BLE001 - one candidate never stops the rest
                for name in FILES:
                    (d / name).unlink(missing_ok=True)
                note = f"candidate {nn(h.n)} skipped: {_reason(exc)}"
                notes.append(note)
                progress(note)
            if done == total:
                progress(f"First listen: {total} window{'' if total == 1 else 's'} in {listening:.0f} s.")
            progress(f"Windows: {done} of {total}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    index = CandidateIndex(language=language or detected or DEFAULT_LANGUAGE,
                           captions=transcribe.captions_allowed(" ".join(texts)),
                           ready=sorted(ready), notes=notes)
    write_json(job.candidates_index, index.to_dict())
    if not ready:
        why = f" ({notes[0]})" if notes else " (there were none)"
        raise StageError("None of the hot spots could be prepared for judging" + why)
    return index


def _transcribe(job: JobDir, h: Hotspot, language: str | None, model) -> tuple[str, str]:
    """Writes the window's transcript; gives its words as one string and the language heard."""
    transcript, heard = transcribe.transcribe_window(job.audio, h.start, h.end,
                                                     language=language, model=model)
    write_json(job.candidate(h.n) / "transcript.json", transcript.to_dict())
    return " ".join(w.text for w in transcript.words), heard
