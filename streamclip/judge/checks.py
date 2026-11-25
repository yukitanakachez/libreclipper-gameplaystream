"""Checking everything the judge said before any of it reaches a clip.

The model answers in window seconds and gets things roughly right. This module turns its
times into stream seconds, snaps them to word edges, makes each stretch a length that
finishes at 15 to 40 seconds, tidies the text fields, drops the weaker of two picks that
show the same moment, and ranks what is left.

A job never returns fewer clips than its floor. When the model kept too few, the best of the
ones it dropped fill the gap, marked `weak`. When the model gave no usable answer at all,
`fallback_picks` ranks the hot spots by excitement and marks every one of them `weak`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from clipper import pacing
from clipper.jobs import read_json
from clipper.selection import GUARD, PREROLL, ends_sentence, snap_bounds
from streamclip import settings
from streamclip.jobdir import JobDir
from streamclip.schema import Beat, Hotspot, Pick, Transcript, Word

BEAT_KINDS = ("payoff", "reaction", "hard_cut")
SAME_MOMENT = 0.5               # two picks sharing more than this much of the shorter are one moment
MAX_SOURCE = settings.MAX_FINISHED * settings.SPEED      # 52 s of stream finishes at 40 s
TITLE_WORDS = 8
MAX_HASHTAGS = 8
NO_TITLE = "Stream highlight"
CONTENT = "content:"            # how the prompt has the model mark a drop that must stay dropped
FALLBACK_BEFORE, FALLBACK_AFTER = 25.0, 10.0     # a cut made without the model, around the peak
UNRATED = "The judge did not rate this moment. It was picked on excitement alone."

_PACE = pacing.Pacing(True, settings.DEAD_AIR, settings.SPEED)


@dataclass
class _Window:
    hotspot: Hotspot
    transcript: Transcript          # stream seconds; `duration` is the window's end


# --- reading the job ------------------------------------------------------------------------

def _transcript(job: JobDir, hotspot: Hotspot) -> Transcript:
    try:
        found = Transcript.from_dict(read_json(job.candidate(hotspot.n) / "transcript.json"))
    except (OSError, ValueError, KeyError, TypeError):
        return Transcript([], [], hotspot.end)
    # The snapping code finds a word's neighbours by its index, and stops at `duration`.
    words = [Word(k, w.text, w.start, w.end, w.prob) for k, w in enumerate(found.words)]
    return Transcript(words, found.segments, hotspot.end)


def _windows(job: JobDir) -> dict[int, _Window]:
    """Every ready candidate, by number."""
    try:
        ready = set(read_json(job.candidates_index)["ready"])
        hotspots = [Hotspot.from_dict(h) for h in read_json(job.hotspots)["hotspots"]]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return {h.n: _Window(h, _transcript(job, h)) for h in hotspots if h.n in ready}


def _stream(job: JobDir) -> dict:
    try:
        return read_json(job.stream)
    except (OSError, ValueError):
        return {}


def _stream_title(stream: dict) -> str:
    return " ".join(str(stream.get("title") or "").split()[:TITLE_WORDS]) or NO_TITLE


# --- times and lengths ----------------------------------------------------------------------

def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _spans(value, offset: float) -> list[tuple[float, float]]:
    """The model's protected stretches, in stream seconds."""
    spans = []
    for item in value if isinstance(value, list) else []:
        pair = (item.get("start"), item.get("end")) if isinstance(item, dict) else item
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = _number(pair[0]), _number(pair[1])
        if a is not None and b is not None and b > a:
            spans.append((offset + a, offset + b))
    return sorted(spans)


def _finished(start: float, end: float, transcript: Transcript,
              protect: list[tuple[float, float]]) -> float:
    """About how long the stretch plays once pauses are cut and it is sped up.

    Only gaps between words are counted as cut, so this is the shortest it can come out. With
    no words nothing is counted as cut. A protected stretch is never cut, so it counts as speech.
    """
    held = [Word(-1, "", a, b) for a, b in protect]
    words = [w for w in [*transcript.words, *held] if w.end > start and w.start < end]
    return pacing.finished_duration(pacing.keep_intervals(words, start, end, _PACE), _PACE)


def _openings(transcript: Transcript) -> list[float]:
    """Where a clip may open: just before each word that begins a sentence.

    Without punctuation the segments' first words stand in, and without those any word."""
    words = transcript.words
    if not words:
        return []
    first = {0} | {i for i in range(1, len(words)) if ends_sentence(words[i - 1])}
    if len(first) == 1:
        first |= {s.first_word for s in transcript.segments if 0 <= s.first_word < len(words)}
    if len(first) == 1:
        first = set(range(len(words)))

    def opening(i: int) -> float:
        t = words[i].start - PREROLL
        if i > 0:
            t = max(t, words[i - 1].end + GUARD)
        return min(t, words[i].start)
    return sorted(opening(i) for i in first)


def _bounds(start: float, end: float, window: _Window,
            protect: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Stream-second bounds that sit in the window, on word edges, at a usable length.

    The ending is the payoff, so only the start ever moves to fix the length. None when no
    start makes the stretch long enough."""
    low, high = window.hotspot.start, window.hotspot.end
    transcript = window.transcript
    start, end = min(max(start, low), high), min(max(end, low), high)
    if end <= start:
        return None
    if transcript.words:
        start, end = snap_bounds(start, end, transcript)
        start, end = max(start, low), min(end, high)
        if end <= start:
            return None
    openings = [t for t in _openings(transcript) if low <= t < end]

    if end - start > MAX_SOURCE:
        earliest = end - MAX_SOURCE
        start = earliest
        later = [t for t in openings if t >= earliest]
        if later and _finished(later[0], end, transcript, protect) >= settings.MIN_FINISHED:
            start = later[0]

    if _finished(start, end, transcript, protect) < settings.MIN_FINISHED:
        if transcript.words:
            earlier = [t for t in reversed(openings) if t < start] + [low]
        else:
            earlier = [end - settings.MIN_FINISHED * settings.SPEED, low]
        for t in earlier:
            if t < low or end - t > MAX_SOURCE:
                break
            start = min(start, t)
            if _finished(start, end, transcript, protect) >= settings.MIN_FINISHED:
                break
        if _finished(start, end, transcript, protect) < settings.MIN_FINISHED:
            return None
    return start, end


# --- tidying the text fields ----------------------------------------------------------------

def _score(value) -> int:
    number = _number(value)
    return 1 if number is None else max(1, min(10, round(number)))


def _text(value) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _hashtags(value) -> list[str]:
    tags, seen = [], set()
    for item in value if isinstance(value, list) else []:
        tag = re.sub(r"\W", "", item) if isinstance(item, str) else ""
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append("#" + tag)
    return tags[:MAX_HASHTAGS]


def _beats(value, offset: float, start: float, end: float) -> list[Beat]:
    beats = []
    for item in value if isinstance(value, list) else []:
        t = _number(item.get("t")) if isinstance(item, dict) else None
        if t is not None and item.get("kind") in BEAT_KINDS and start <= offset + t <= end:
            beats.append(Beat(item["kind"], offset + t))
    return sorted(beats, key=lambda beat: beat.t)


def _pick(row: dict, window: _Window, stream: dict, weak: bool) -> Pick | None:
    """One answer row as a Pick, or None when its times cannot make a clip."""
    offset = window.hotspot.start
    start, end = _number(row.get("start")), _number(row.get("end"))
    if start is None or end is None:
        return None
    protect = _spans(row.get("protect"), offset)
    bounds = _bounds(offset + start, offset + end, window, protect)
    if bounds is None:
        return None
    start, end = bounds
    return Pick(
        rank=0, candidate=window.hotspot.n, start=start, end=end, score=_score(row.get("score")),
        reason=_text(row.get("reason")),
        title=" ".join(_text(row.get("title")).split()[:TITLE_WORDS]) or _stream_title(stream),
        caption=_text(row.get("caption")), hashtags=_hashtags(row.get("hashtags")),
        beats=_beats(row.get("beats"), offset, start, end),
        protect=[(max(a, start), min(b, end)) for a, b in protect if min(b, end) > max(a, start)],
        weak=weak)


# --- ranking --------------------------------------------------------------------------------

def _same_moment(a: Pick, b: Pick) -> bool:
    shared = min(a.end, b.end) - max(a.start, b.start)
    return shared > SAME_MOMENT * min(a.end - a.start, b.end - b.start)


def _distinct(picks: list[Pick], taken: list[Pick] = ()) -> list[Pick]:
    """The picks in the order given, without any that repeat an earlier or a taken moment."""
    kept: list[Pick] = []
    for pick in picks:
        if not any(_same_moment(pick, other) for other in [*taken, *kept]):
            kept.append(pick)
    return kept


def _best_first(picks: list[Pick]) -> list[Pick]:
    """Best score first, earlier first on a tie, keeping one pick per moment."""
    return _distinct(sorted(picks, key=lambda p: (-p.score, p.start)))


def _numbered(picks: list[Pick]) -> list[Pick]:
    for rank, pick in enumerate(picks, start=1):
        pick.rank = rank
    return picks


def _rows(raw: dict) -> dict[int, dict]:
    """The answer's rows by candidate number, the first one given for each."""
    rows: dict[int, dict] = {}
    listed = raw.get("candidates") if isinstance(raw, dict) else None
    for row in listed if isinstance(listed, list) else []:
        n = row.get("n") if isinstance(row, dict) else None
        if isinstance(n, int) and not isinstance(n, bool):
            rows.setdefault(n, row)
    return rows


# --- the floor and the fallback -------------------------------------------------------------

def floor_for(duration: float, wanted: int) -> int:
    """The fewest clips a stream of this length returns: five, or one per eight minutes when
    that is fewer, and never more than were asked for."""
    return min(wanted, settings.FLOOR, max(1, int(duration // settings.SHORT_STREAM_PER_CLIP)))


def _floor(stream: dict, wanted: int) -> int:
    duration = _number(stream.get("duration"))
    return floor_for(settings.SHORT_STREAM if duration is None else duration, wanted)


def _around_peak(window: _Window, stream: dict, score: int, reason: str) -> Pick | None:
    """A weak pick cut without the model: 25 s before the peak to 10 s after it.

    It has no title, since the stream's name (a file's name, for a file) is not one, so no
    title sticker is drawn. Its one beat is a payoff at the peak, which gives the clip its
    sound and its zoom there."""
    spot = window.hotspot
    length = FALLBACK_BEFORE + FALLBACK_AFTER
    start = max(spot.start, spot.peak - FALLBACK_BEFORE)
    end = min(spot.end, max(spot.peak + FALLBACK_AFTER, start + length))
    start = max(spot.start, min(start, end - length))
    bounds = _bounds(start, end, window, [])
    if bounds is None:
        return None
    beats = [Beat("payoff", spot.peak)] if bounds[0] <= spot.peak <= bounds[1] else []
    return Pick(rank=0, candidate=spot.n, start=bounds[0], end=bounds[1], score=score,
                reason=reason, title="", caption="", hashtags=[], beats=beats,
                protect=[], weak=True)


def _what_fired(spot: Hotspot) -> str:
    names = list(dict.fromkeys(f.detail or f.signal for f in sorted(spot.fired, key=lambda f: f.t)))
    return f" ({', '.join(names[:4])})" if names else ""


def _fillers(rows: dict[int, dict], windows: dict[int, _Window], stream: dict) -> list[Pick]:
    """Weak picks to fill the floor with, best first.

    The model's own dropped candidates come first, by its score, in the stretch it chose.
    Ones it dropped for content never come back. Candidates it said nothing about come last,
    by excitement."""
    dropped = sorted(((n, row) for n, row in rows.items()
                      if n in windows and row.get("keep") is not True
                      and not _text(row.get("reason")).lower().startswith(CONTENT)),
                     key=lambda item: (-_score(item[1].get("score")), item[0]))
    fillers = []
    for n, row in dropped:
        pick = _pick(row, windows[n], stream, weak=True) or _around_peak(
            windows[n], stream, _score(row.get("score")), _text(row.get("reason")))
        if pick is not None:
            fillers.append(pick)
    unrated = sorted((w for n, w in windows.items() if n not in rows),
                     key=lambda w: (-w.hotspot.score, w.hotspot.peak))
    for window in unrated:
        pick = _around_peak(window, stream, 1, UNRATED)
        if pick is not None:
            fillers.append(pick)
    return fillers


def apply_checks(raw: dict, job: JobDir, wanted: int) -> list[Pick]:
    """The judge's raw answer as checked picks, best first, at most `wanted` of them.

    When fewer than the floor were kept, weak picks fill up to the floor and no further."""
    windows = _windows(job)
    stream = _stream(job)
    rows = _rows(raw)
    kept = [pick for n, row in rows.items()
            if n in windows and row.get("keep") is True
            and (pick := _pick(row, windows[n], stream, weak=False)) is not None]
    picks = _best_first(kept)[:wanted]
    short = _floor(stream, wanted) - len(picks)
    if short > 0:
        picks += _distinct(_fillers(rows, windows, stream), taken=picks)[:short]
    return _numbered(picks)


def ruled_out(raw: dict) -> set[int]:
    """The candidates an answer dropped for their content. They never come back as clips."""
    return {n for n, row in _rows(raw).items()
            if row.get("keep") is not True and _text(row.get("reason")).lower().startswith(CONTENT)}


def fallback_picks(job: JobDir, wanted: int, skip: set[int] | frozenset[int] = frozenset()) -> list[Pick]:
    """Picks made with no model answer: the hot spots by excitement, every one marked weak.

    Weak clips only ever fill the floor, so this returns the floor's number of picks, or
    fewer when fewer candidates are ready. Candidates in `skip` are left out."""
    stream = _stream(job)
    hottest = sorted((w for n, w in _windows(job).items() if n not in skip),
                     key=lambda w: (-w.hotspot.score, w.hotspot.peak))
    picks = []
    for window in hottest:
        score = max(1, min(10, round((_number(window.hotspot.score) or 0.0) * 10)))
        pick = _around_peak(window, stream, score, "No judge answer. Picked on excitement alone"
                            + _what_fired(window.hotspot) + ".")
        if pick is not None:
            picks.append(pick)
    return _numbered(_distinct(picks)[:_floor(stream, wanted)])
