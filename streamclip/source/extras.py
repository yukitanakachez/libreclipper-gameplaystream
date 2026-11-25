"""The three Twitch extras: chat times, viewer clips and the creator's picture.

Each is optional. None ever fails a job: `optional` turns a timeout or any error into
"missing: <why>". No key, no login and no cookies are used anywhere:

- Chat comes from TwitchDownloaderCLI (`chatdownload`), found on PATH or at
  $STREAMCLIP_TWITCHDOWNLOADER. yt-dlp dropped chat in 2026.06.
- Viewer clips come from Twitch's public web GraphQL, the interface twitch.tv itself and
  yt-dlp call, sending only the website's public Client-ID. yt-dlp's clip listing gives no
  broadcast or offset, so clips could not be matched to a broadcast through it.
- The picture comes from the `og:image` tag of the channel page, with GraphQL as a fallback.

Parsing is kept apart from fetching so each piece is tested on canned answers."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import ssl
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime
from html import unescape
from pathlib import Path

from clipper import shell
from streamclip.schema import ViewerClip
from streamclip.source import media

EXTRA_TIMEOUT = 300.0                       # seconds each extra may take before the job goes on
HTTP_TIMEOUT = 30.0
GQL_URL = "https://gql.twitch.tv/gql"
WEB_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"     # twitch.tv's own public web client
USER_AGENT = "Mozilla/5.0 (compatible; streamclip)"
CLIP_PAGES = 10                             # at most this many pages of 100 clips per sort
# Twitch serves the first page of each order to anyone, but every later page asks for a
# browser integrity token (checked 2026-11-23), so two orders are read to see more clips.
CLIP_SORTS = ("VIEWS_DESC", "TRENDING")
RETRY_WAIT = 1.0                            # seconds before a page that failed is asked again
TOOL_ENV = "STREAMCLIP_TWITCHDOWNLOADER"
TOOL_NAME = "TwitchDownloaderCLI"

Fetch = Callable[..., bytes]                # fetch(url, data=None, headers=None) -> bytes


class Missing(Exception):
    """An extra is unavailable for a known reason; str(e) is that reason."""


def optional(fn: Callable[[], object], name: str) -> tuple[object | None, str]:
    """Run `fn` in a worker thread for at most EXTRA_TIMEOUT seconds.

    Gives (value, "ok"), or (None, "missing: <why>") on a timeout or any exception. A thread
    that runs past the wait is left to finish on its own (it is a daemon thread, and every
    command it runs carries its own time limit)."""
    box: dict[str, object] = {}

    def work() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:            # noqa: BLE001 - an extra never fails a job
            box["error"] = exc

    wait = EXTRA_TIMEOUT
    worker = threading.Thread(target=work, name=f"extra-{name}", daemon=True)
    worker.start()
    worker.join(wait)
    if worker.is_alive():
        return None, f"missing: {name} timed out after {wait:g} s"
    if "error" in box:
        return None, f"missing: {_why(box['error'], name)}"
    return box.get("value"), "ok"


def _why(exc: BaseException, name: str) -> str:
    if isinstance(exc, Missing):
        return str(exc)
    if isinstance(exc, shell.CommandError):
        detail = shell.describe_failure(exc)
    else:
        detail = str(exc).strip() or type(exc).__name__
    return f"{name} failed: {detail[:200]}"


# --- fetching over HTTP --------------------------------------------------------------------

def http_get(url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
    """GET `url`, or POST `data` to it. Raises on any HTTP error."""
    context = None
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:                         # pragma: no cover - certifi is a dependency
        pass
    request = urllib.request.Request(url, data=data,
                                     headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT, context=context) as answer:
        return answer.read()


def gql(query: str, variables: dict, fetch: Fetch) -> dict:
    """One query to Twitch's public web GraphQL. Raises Missing when it answers with errors."""
    body = json.dumps({"query": query, "variables": variables}).encode()
    answer = json.loads(fetch(GQL_URL, data=body, headers={
        "Client-ID": WEB_CLIENT_ID, "Content-Type": "application/json"}))
    if isinstance(answer, list):
        answer = answer[0] if answer else {}
    if answer.get("errors"):
        messages = "; ".join(str(e.get("message", e)) for e in answer["errors"])
        raise Missing(f"Twitch answered with an error: {messages[:200]}")
    return answer


# --- chat ----------------------------------------------------------------------------------

def chat_tool() -> str | None:
    """TwitchDownloaderCLI: $STREAMCLIP_TWITCHDOWNLOADER if it points at a file, else PATH."""
    configured = os.environ.get(TOOL_ENV)
    if configured and Path(configured).is_file():
        return configured
    return shutil.which(TOOL_NAME)


def parse_chat(raw: dict) -> list[float]:
    """Message times in stream seconds, sorted, from TwitchDownloaderCLI's chat JSON."""
    times = []
    for comment in raw.get("comments") or []:
        try:
            t = float(comment.get("content_offset_seconds"))
        except (TypeError, ValueError, AttributeError):
            continue
        if t >= 0:
            times.append(t)
    return sorted(times)


def download_chat(video_id: str, run=shell.run, tool: str | None = None) -> list[float]:
    tool = tool or chat_tool()
    if not tool:
        raise Missing("chat tool not installed")
    # One thread: with its default of four the tool drops most of the chat and still exits
    # 0 (measured 2026-11-23 on a 5-hour broadcast: 5,568 messages against 56,135, the
    # one-thread run taking 149 s).
    with tempfile.TemporaryDirectory(prefix="streamclip-chat-") as work:
        out = Path(work) / "chat.json"
        run([tool, "chatdownload", "--id", video_id, "-o", str(out), "--threads", "1",
             "--collision", "Overwrite", "--temp-path", work,
             "--banner", "false", "--log-level", "Error"], timeout=int(EXTRA_TIMEOUT) + 60)
        if not out.is_file():
            raise Missing("chat tool wrote no file")
        return parse_chat(json.loads(out.read_text(encoding="utf-8")))


# --- viewer clips --------------------------------------------------------------------------

CLIPS_QUERY = """
query ($login: String!, $period: ClipsPeriod, $sort: ClipsSort, $after: Cursor) {
  user(login: $login) {
    clips(first: 100, after: $after, criteria: {period: $period, sort: $sort}) {
      pageInfo { hasNextPage }
      edges { cursor node { slug title viewCount durationSeconds videoOffsetSeconds video { id } } }
    }
  }
}"""


def clip_period(started: float | None, now: float | None = None) -> str:
    """The smallest of Twitch's clip periods that reaches back to the broadcast's start."""
    if not started:
        return "ALL_TIME"
    age = (now if now is not None else time.time()) - started
    for limit, period in ((86400, "LAST_DAY"), (7 * 86400, "LAST_WEEK"),
                          (30 * 86400, "LAST_MONTH")):
        if age < limit - 3600:                  # an hour's margin for clocks and slow clips
            return period
    return "ALL_TIME"


def _clips(answer: dict) -> dict:
    return (((answer.get("data") or {}).get("user") or {}).get("clips")) or {}


def parse_viewer_clips(answer: dict, video_id: str) -> list[ViewerClip]:
    """The clips on one page of answers that were cut from this broadcast, by offset."""
    found = []
    for edge in _clips(answer).get("edges") or []:
        node = (edge or {}).get("node") or {}
        if str((node.get("video") or {}).get("id") or "") != str(video_id):
            continue
        if node.get("videoOffsetSeconds") is None:
            continue
        found.append(ViewerClip(offset=float(node["videoOffsetSeconds"]),
                                duration=float(node.get("durationSeconds") or 0.0),
                                views=int(node.get("viewCount") or 0),
                                title=str(node.get("title") or "")))
    return sorted(found, key=lambda c: c.offset)


def _gql_again(query: str, variables: dict, fetch: Fetch) -> dict:
    """gql, asked once more after a failure. Twitch's web GraphQL often answers "service
    error" and then succeeds. An integrity refusal is final, so it is not asked again."""
    try:
        return gql(query, variables, fetch)
    except Exception as exc:                    # noqa: BLE001
        if "integrity" in str(exc).lower():
            raise
        time.sleep(RETRY_WAIT)
        return gql(query, variables, fetch)


def fetch_viewer_clips(login: str, video_id: str, started: float | None, fetch: Fetch,
                       pages: int = CLIP_PAGES) -> list[ViewerClip]:
    """This broadcast's viewer clips, from each order in CLIP_SORTS, each clip once.

    A refused page ends that order and keeps what came before it. Only when no order gave
    a single page does this raise."""
    if not login:
        raise Missing("viewer clips need the creator's name")
    found: dict[tuple, ViewerClip] = {}
    answered, last_error = False, None
    for sort in CLIP_SORTS:
        variables: dict = {"login": login, "period": clip_period(started), "sort": sort}
        for _ in range(pages):
            try:
                answer = _gql_again(CLIPS_QUERY, variables, fetch)
            except Exception as exc:            # noqa: BLE001 - keep what earlier pages gave
                last_error = exc
                break
            answered = True
            for clip in parse_viewer_clips(answer, video_id):
                found.setdefault((clip.offset, clip.duration, clip.views, clip.title), clip)
            clips = _clips(answer)
            edges = clips.get("edges") or []
            cursor = next((e.get("cursor") for e in reversed(edges) if e and e.get("cursor")), None)
            if not (clips.get("pageInfo") or {}).get("hasNextPage") or not cursor:
                break
            variables = {**variables, "after": cursor}
    if not answered and last_error is not None:
        raise last_error
    return sorted(found.values(), key=lambda c: c.offset)


# --- the creator's picture -----------------------------------------------------------------

_OG_IMAGE = re.compile(
    r"""<meta\s+[^>]*?(?:property|name)\s*=\s*["']og:image["'][^>]*?>""", re.I)
_CONTENT = re.compile(r"""content\s*=\s*["']([^"']+)["']""", re.I)

PICTURE_QUERY = "query ($login: String!) { user(login: $login) { profileImageURL(width: 300) } }"


def parse_og_image(html: str) -> str | None:
    tag = _OG_IMAGE.search(html or "")
    content = _CONTENT.search(tag.group(0)) if tag else None
    return unescape(content.group(1)) if content else None


def picture_url(login: str, fetch: Fetch) -> str:
    try:
        url = parse_og_image(fetch(f"https://www.twitch.tv/{login}").decode("utf-8", "replace"))
    except Exception:                           # noqa: BLE001 - fall back to GraphQL
        url = None
    if url and url.startswith("https://"):
        return url
    answer = gql(PICTURE_QUERY, {"login": login}, fetch)
    url = ((answer.get("data") or {}).get("user") or {}).get("profileImageURL")
    if not url:
        raise Missing("Twitch gave no picture for that creator")
    return url


def save_jpeg(image: bytes, dest: Path) -> Path:
    """Any picture Pillow can read, as an RGB JPEG; see-through parts go white."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(image)) as img:
            img.load()
            if img.mode in ("RGBA", "LA", "P"):
                rgba = img.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.getchannel("A"))
            else:
                flat = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise Missing("the creator's picture could not be read") from exc
    with media.partial(dest) as tmp:
        flat.save(tmp, "JPEG", quality=92)
    return Path(dest)


def fetch_avatar(login: str, dest: Path, fetch: Fetch) -> Path:
    if not login:
        raise Missing("no creator name to look up")
    return save_jpeg(fetch(picture_url(login, fetch)), dest)


def started_epoch(started_at: str) -> float | None:
    try:
        return datetime.fromisoformat(started_at).timestamp() if started_at else None
    except ValueError:
        return None
