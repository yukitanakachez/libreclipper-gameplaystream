"""The judge's one model call. Standard library only.

On Starwish this very file runs as a tool on a machine that has Python and nothing else, so
it imports nothing from the engine and reads the job folder's JSON by hand:

    python3 call.py <job_root> <out_dir>      writes <out_dir>/judge_raw.json

Exit codes: 0 the answer was written, 2 usage error, 3 the answer was unusable twice,
4 the key was missing or rejected. `prompt.md` and `answer_schema.json` sit beside this file.

Every time sent to the model and every time in its answer counts from the start of that
candidate's own window. `checks.py` turns them into stream seconds.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "claude-opus-5-5"
URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2026-10-21"
# If the model's safety check declines the request, the service re-runs it on its fallback.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 64000              # thinking and the answer share this
EFFORT = "high"
BATCH = 24                      # candidates in one request
TIMEOUT = 900                   # seconds without a byte before the call is given up
RETRY_WAIT = 5.0                # seconds before the one retry
LINE_WORDS = 14                 # a transcript line is broken after this many words
KEY_NAMES = ("JUDGE_ANTHROPIC_KEY", "ANTHROPIC_API_KEY")
OUT_NAME = "judge_raw.json"
SENTENCE_END = (".", "?", "!")
HERE = Path(__file__).resolve().parent


class JudgeError(Exception):
    """The model gave no usable answer. str(e) is a plain reason."""


class KeyRejected(JudgeError):
    """The service did not accept the key. Asking again would not help."""


# --- reading the job folder ---------------------------------------------------------------

def _read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nn(n: int) -> str:
    return f"{n:02d}"


def _ready(job_root: Path) -> list[dict]:
    """The hot spots whose candidates are ready, in number order."""
    ready = set(_read(job_root / "candidates" / "index.json").get("ready", []))
    hotspots = {h["n"]: h for h in _read(job_root / "hotspots.json").get("hotspots", [])}
    return [hotspots[n] for n in sorted(ready) if n in hotspots]


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# --- what the model is shown --------------------------------------------------------------

def _transcript_lines(transcript: dict, window_start: float) -> list[str]:
    """`[61.0] OH MY GOD`: a line per sentence or segment, timed in window seconds."""
    words = transcript.get("words") or []
    breaks = {s.get("first_word") for s in transcript.get("segments") or []}
    lines: list[list[dict]] = []
    for k, word in enumerate(words):
        last = lines[-1] if lines else None
        if (last is None or word.get("i", k) in breaks or len(last) >= LINE_WORDS
                or str(last[-1]["text"]).rstrip("\"')").endswith(SENTENCE_END)):
            lines.append([])
        lines[-1].append(word)
    return [f"[{max(0.0, line[0]['start'] - window_start):.1f}] "
            + " ".join(str(w["text"]).strip() for w in line) for line in lines]


def _fired(hotspot: dict) -> str:
    fired = sorted(hotspot.get("fired") or [], key=lambda f: f["t"])
    if not fired:
        return "(nothing recorded)"
    return ", ".join(f"{f.get('detail') or f['signal']} at {max(0.0, f['t'] - hotspot['start']):.0f} s"
                     for f in fired)


def _game_at(chapters: list[dict], t: float) -> str:
    return next((str(c["title"]) for c in chapters if c["start"] <= t < c["end"] and c.get("title")), "")


def _candidate_blocks(job_root: Path, hotspot: dict, chapters: list[dict]) -> list[dict]:
    folder = job_root / "candidates" / _nn(hotspot["n"])
    start = hotspot["start"]
    try:
        transcript = _read(folder / "transcript.json")
    except (OSError, ValueError):
        transcript = {}
    lines = _transcript_lines(transcript, start) or ["(no speech in this window)"]
    game = _game_at(chapters, hotspot["peak"])
    sheet = folder / "sheet.jpg"
    head = [f"Candidate {hotspot['n']}",
            f"Window: {hotspot['end'] - start:.0f} s long, {_clock(start)} into the stream"
            + (f", playing {game}." if game else "."),
            f"What fired: {_fired(hotspot)}",
            "Transcript:", *lines,
            "Frames: the image that follows." if sheet.is_file()
            else "Frames: (no frames for this candidate)"]
    blocks: list[dict] = [{"type": "text", "text": "\n".join(head)}]
    if sheet.is_file():
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(sheet.read_bytes()).decode("ascii")}})
    return blocks


def build_request(job_root: Path, model: str = MODEL, only: list[int] | None = None) -> dict:
    """The JSON body of one request. `only` limits it to those candidate numbers."""
    job_root = Path(job_root)
    hotspots = [h for h in _ready(job_root) if only is None or h["n"] in only]
    try:
        stream = _read(job_root / "stream.json")
    except (OSError, ValueError):
        stream = {}
    chapters = stream.get("chapters") or []
    games = list(dict.fromkeys(str(c["title"]) for c in chapters if c.get("title")))
    intro = [f"Stream: {stream.get('title') or '(untitled)'}",
             f"Creator: {stream.get('creator_name') or stream.get('creator_login') or '(unknown)'}",
             f"Length: {_clock(stream.get('duration') or 0)}"]
    if games:
        intro.append("Games: " + ", ".join(games[:12]))
    intro.append(f"{len(hotspots)} candidates follow, numbered "
                 + ", ".join(str(h["n"]) for h in hotspots) + ".")
    content: list[dict] = [{"type": "text", "text": "\n".join(intro)}]
    for hotspot in hotspots:
        content += _candidate_blocks(job_root, hotspot, chapters)
    content.append({"type": "text", "text":
                    f"Judge all {len(hotspots)} candidates above against each other and return "
                    "one entry for each, with every time in that candidate's window seconds."})
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "system": (HERE / "prompt.md").read_text(encoding="utf-8"),
        "output_config": {"effort": EFFORT, "format": {
            "type": "json_schema", "schema": _read(HERE / "answer_schema.json")}},
        "fallbacks": "default",
        "messages": [{"role": "user", "content": content}],
    }


# --- the call -----------------------------------------------------------------------------

def _post(url: str, headers: dict, body: bytes) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _headers(key: str) -> dict:
    return {"x-api-key": key, "anthropic-version": API_VERSION, "anthropic-beta": FALLBACK_BETA,
            "content-type": "application/json"}


def _explain(raw: bytes) -> str:
    """The service's own error message, shortened."""
    try:
        body = json.loads(raw)
        detail = body["error"]["message"]
    except (ValueError, KeyError, TypeError):
        detail = raw.decode("utf-8", "replace")
    return " ".join(str(detail).split())[:160]


def _message(raw: bytes) -> dict:
    """The reply as one message, whether it came whole or as a stream of events."""
    text = raw.decode("utf-8", "replace")
    if text.lstrip().startswith("{"):
        try:
            message = json.loads(text)
        except ValueError as exc:
            raise JudgeError("the reply was not JSON") from exc
        if not isinstance(message, dict):
            raise JudgeError("the reply was not a message")
        return message
    message: dict = {"model": "", "stop_reason": None, "content": [], "usage": {}}
    blocks: dict[int, dict] = {}
    seen = False
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:])
        except ValueError:
            continue
        seen = True
        kind = event.get("type")
        if kind == "message_start":
            message["model"] = event.get("message", {}).get("model", "")
            message["usage"].update(event.get("message", {}).get("usage") or {})
        elif kind == "content_block_start":
            blocks[event["index"]] = dict(event.get("content_block") or {})
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and event.get("index") in blocks:
                block = blocks[event["index"]]
                block["text"] = block.get("text", "") + delta.get("text", "")
        elif kind == "message_delta":
            message["stop_reason"] = (event.get("delta") or {}).get("stop_reason")
            message["usage"].update(event.get("usage") or {})       # the final output count
        elif kind == "error":
            raise JudgeError("the service stopped mid-answer: "
                             + _explain(json.dumps(event).encode()))
    if not seen:
        raise JudgeError("the reply was not JSON")
    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


def _verdicts(message: dict, asked: set[int]) -> list[dict]:
    """The answer's entries for the candidates that were asked about."""
    stop = message.get("stop_reason")
    if stop == "refusal":
        raise JudgeError("the model refused to judge these candidates")
    if stop == "max_tokens":
        raise JudgeError("the model ran out of room before finishing its answer")
    if stop != "end_turn":
        raise JudgeError("the answer was cut short")
    texts = [b.get("text", "") for b in message.get("content") or []
             if isinstance(b, dict) and b.get("type") == "text"]
    text = texts[-1] if texts else ""
    first, last = text.find("{"), text.rfind("}")
    try:
        answer = json.loads(text[first:last + 1]) if first != -1 and last > first else None
    except ValueError:
        answer = None
    rows = answer.get("candidates") if isinstance(answer, dict) else None
    if not isinstance(rows, list):
        raise JudgeError("the model's answer was not the JSON asked for")
    kept: dict[int, dict] = {}
    for row in rows:
        n = row.get("n") if isinstance(row, dict) else None
        if isinstance(n, int) and not isinstance(n, bool) and n in asked:
            kept.setdefault(n, row)
    if not kept:
        raise JudgeError("the model's answer covered none of the candidates")
    return list(kept.values())


def _count(usage: dict, message: dict) -> None:
    """Add one reply's tokens to `usage`. A reply that was unusable was still paid for."""
    got = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    usage["requests"] += 1
    for name in ("input_tokens", "output_tokens"):
        value = got.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            usage[name] += value


def _ask(body: dict, asked: set[int], key: str, post, usage: dict) -> tuple[list[dict], str]:
    try:
        status, raw = post(URL, _headers(key), json.dumps(body).encode("utf-8"))
    except OSError as exc:
        raise JudgeError(f"the model service could not be reached ({type(exc).__name__})") from exc
    if status in (401, 403):
        raise KeyRejected(f"the model service rejected the key ({status})")
    if status != 200:
        raise JudgeError(f"the model service answered {status}: {_explain(raw)}".replace(key, "***"))
    message = _message(raw)
    _count(usage, message)
    return _verdicts(message, asked), str(message.get("model") or body["model"])


def judge(job_root: Path, key: str, model: str = MODEL, post=None, progress=None) -> dict:
    """Ask the model about every ready candidate. Returns the raw answer:

        {"model": str, "candidates": [{"n", "keep", "score", "reason", "start", "end", "title",
                                       "caption", "hashtags", "beats", "protect"}, ...],
         "usage": {"input_tokens": int, "output_tokens": int, "requests": int}}

    `usage` adds up every reply the service sent back, retries included: what the call cost.

    A failed call or an unusable answer is tried once more, then raises JudgeError. A rejected
    key raises KeyRejected at once. A candidate the model left out is simply absent.
    """
    job_root = Path(job_root)
    post = post or _post
    say = progress or (lambda line: None)
    numbers = [h["n"] for h in _ready(job_root)]
    batches = [numbers[i:i + BATCH] for i in range(0, len(numbers), BATCH)]
    say(f"judge: {len(numbers)} candidates in {len(batches)} request" + ("" if len(batches) == 1 else "s"))
    rows: list[dict] = []
    served = model
    usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    for k, batch in enumerate(batches, start=1):
        body = build_request(job_root, model, only=batch)
        for attempt in (1, 2):
            say(f"judge: asking {model}, request {k} of {len(batches)}"
                + (", second try" if attempt == 2 else ""))
            try:
                answered, served = _ask(body, set(batch), key, post, usage)
                break
            except KeyRejected:
                raise
            except JudgeError as exc:
                say(f"judge: {exc}")
                if attempt == 2:
                    raise
                time.sleep(RETRY_WAIT)
        rows += answered
    say(f"judge: {usage['requests']} request{'' if usage['requests'] == 1 else 's'} used "
        f"{usage['input_tokens']} input tokens and {usage['output_tokens']} output tokens")
    return {"model": served, "candidates": sorted(rows, key=lambda row: row["n"]), "usage": usage}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 2:
        print("usage: python3 call.py <job_root> <out_dir>", file=sys.stderr)
        return 2
    job_root, out_dir = Path(args[0]), Path(args[1])
    if not (job_root / "candidates" / "index.json").is_file() or not (job_root / "hotspots.json").is_file():
        print(f"judge: {job_root} holds no candidates to judge", file=sys.stderr)
        return 2
    key = next((os.environ[name].strip() for name in KEY_NAMES if os.environ.get(name, "").strip()), "")
    if not key:
        print(f"judge: no model key is set (expected {KEY_NAMES[0]})", file=sys.stderr)
        return 4
    try:
        raw = judge(job_root, key, progress=print)
    except KeyRejected as exc:
        print(f"judge: {exc}", file=sys.stderr)
        return 4
    except JudgeError:
        print("judge: no usable answer after two tries", file=sys.stderr)
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / (OUT_NAME + ".partial")
    partial.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, out_dir / OUT_NAME)
    kept = sum(1 for row in raw["candidates"] if row.get("keep") is True)
    print(f"judge: wrote {OUT_NAME}, {kept} of {len(raw['candidates'])} kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
