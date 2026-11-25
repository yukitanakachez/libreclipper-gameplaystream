"""The three optional extras: chat, viewer clips and the creator's picture. Every one is fed
canned answers; none touches the network."""

import io
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from clipper import shell
from streamclip.schema import ViewerClip
from streamclip.source import extras
from streamclip.source.local import FileSource
from streamclip.source.twitch import TwitchSource

FIXTURES = Path(__file__).parent / "fixtures" / "twitch"
LINK = "https://www.twitch.tv/videos/123"


def fake_run(answer=None, fail=None):
    def run(cmd, timeout=3600):
        if fail:
            raise shell.CommandError(cmd, 1, fail)
        return answer
    return run


@pytest.fixture
def vod_json() -> str:
    return (FIXTURES / "vod.json").read_text()


# --- optional() --------------------------------------------------------------------------

def test_slow_extra_times_out(monkeypatch):
    monkeypatch.setattr(extras, "EXTRA_TIMEOUT", 0.2)
    value, note = extras.optional(lambda: time.sleep(5), "chat")
    assert value is None and note == "missing: chat timed out after 0.2 s"


def test_failing_extra_never_raises():
    value, note = extras.optional(lambda: 1 / 0, "viewer clips")
    assert value is None and note.startswith("missing: ")


def test_a_known_reason_is_passed_on():
    def fn():
        raise extras.Missing("chat tool not installed")
    assert extras.optional(fn, "chat") == (None, "missing: chat tool not installed")


def test_a_working_extra_is_ok():
    assert extras.optional(lambda: [1.0], "chat") == ([1.0], "ok")


# --- chat --------------------------------------------------------------------------------

def test_chat_json_becomes_sorted_times(tmp_path):
    raw = {"comments": [{"content_offset_seconds": 12.5}, {"content_offset_seconds": 3.0}]}
    assert extras.parse_chat(raw) == [3.0, 12.5]


def test_chat_skips_broken_comments():
    raw = {"comments": [{"content_offset_seconds": "7"}, {}, {"content_offset_seconds": None},
                        {"content_offset_seconds": -1}]}
    assert extras.parse_chat(raw) == [7.0]


def test_missing_tool_is_a_reason_not_an_error(monkeypatch):
    monkeypatch.setattr(extras, "chat_tool", lambda: None)
    assert TwitchSource(LINK, run=fake_run("{}")).chat_times() == (None, "missing: chat tool not installed")


def test_chat_tool_is_found_by_its_setting(monkeypatch, tmp_path):
    tool = tmp_path / "TwitchDownloaderCLI"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setenv("STREAMCLIP_TWITCHDOWNLOADER", str(tool))
    assert extras.chat_tool() == str(tool)
    monkeypatch.setenv("STREAMCLIP_TWITCHDOWNLOADER", str(tmp_path / "nothing"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert extras.chat_tool() is None


def test_chat_runs_the_tool_and_reads_its_file(monkeypatch, vod_json):
    monkeypatch.setattr(extras, "chat_tool", lambda: "/opt/TwitchDownloaderCLI")
    calls = []

    def run(cmd, timeout=3600):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if cmd[0] == "/opt/TwitchDownloaderCLI":
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_text(json.dumps({"comments": [{"content_offset_seconds": 9},
                                                    {"content_offset_seconds": 2}]}))
            return ""
        return vod_json
    assert TwitchSource(LINK, run=run).chat_times() == ([2.0, 9.0], "ok")
    cmd = calls[-1]
    assert cmd[1] == "chatdownload" and cmd[cmd.index("--id") + 1] == "123"
    # With more threads the tool drops most of the chat and still says it succeeded
    # (measured 2026-11-23: 5,568 messages with 4 threads, 56,135 with one).
    assert cmd[cmd.index("--threads") + 1] == "1"
    assert not Path(cmd[cmd.index("-o") + 1]).exists()      # the work folder is cleaned up


def test_chat_tool_failure_is_missing(monkeypatch):
    monkeypatch.setattr(extras, "chat_tool", lambda: "/opt/TwitchDownloaderCLI")
    value, note = TwitchSource(LINK, run=fake_run(fail="[ERROR] - 403")).chat_times()
    assert value is None and note.startswith("missing: ")


# --- viewer clips ------------------------------------------------------------------------

def clip_node(slug, video, offset, views, duration=30, title="t"):
    return {"cursor": slug, "node": {"slug": slug, "title": title, "viewCount": views,
                                     "durationSeconds": duration, "videoOffsetSeconds": offset,
                                     "video": None if video is None else {"id": video}}}


def clips_page(edges, more=False):
    return {"data": {"user": {"clips": {"pageInfo": {"hasNextPage": more}, "edges": edges}}}}


def test_viewer_clips_keep_only_this_broadcast():
    page = clips_page([clip_node("a", "123", 4215, 88960, 5, "nude"),
                       clip_node("b", "999", 10, 50000),
                       clip_node("c", "123", 912, 15665, 14, "first time"),
                       clip_node("d", None, 50, 10),
                       clip_node("e", "123", None, 10)])
    assert extras.parse_viewer_clips(page, "123") == [
        ViewerClip(offset=912.0, duration=14.0, views=15665, title="first time"),
        ViewerClip(offset=4215.0, duration=5.0, views=88960, title="nude")]


class FakeWeb:
    """Answers GraphQL posts with canned pages, in order, and pages by URL."""

    def __init__(self, gql=(), pages=None):
        self.gql, self.pages, self.posts, self.gets = list(gql), pages or {}, [], []

    def __call__(self, url, data=None, headers=None):
        if data is not None:
            self.posts.append((url, json.loads(data), headers))
            return json.dumps(self.gql.pop(0)).encode()
        self.gets.append(url)
        if url not in self.pages:
            raise OSError(f"404 {url}")
        return self.pages[url]


def test_viewer_clips_are_paged_and_matched(vod_json):
    web = FakeWeb(gql=[clips_page([clip_node("a", "123", 100, 5)], more=True),
                       clips_page([clip_node("b", "123", 50, 9), clip_node("c", "7", 1, 1)]),
                       clips_page([clip_node("a", "123", 100, 5), clip_node("t", "123", 70, 2)])])
    clips, note = TwitchSource(LINK, run=fake_run(vod_json), fetch=web).viewer_clips()
    assert note == "ok" and [c.offset for c in clips] == [50.0, 70.0, 100.0]   # "a" once
    url, body, headers = web.posts[0]
    assert url == "https://gql.twitch.tv/gql" and "Client-ID" in headers
    assert "Authorization" not in headers                          # no key, no login
    assert body["variables"]["login"] == "somestreamer" and "after" not in body["variables"]
    assert web.posts[1][1]["variables"]["after"] == "a"
    assert [p[1]["variables"]["sort"] for p in web.posts] == ["VIEWS_DESC", "VIEWS_DESC", "TRENDING"]


def test_a_refused_later_page_keeps_the_first(vod_json, monkeypatch):
    monkeypatch.setattr(extras, "RETRY_WAIT", 0.0)
    # Twitch asks for a browser integrity token on every page after the first.
    web = FakeWeb(gql=[clips_page([clip_node("a", "123", 100, 5)], more=True),
                       {"errors": [{"message": "failed integrity check"}]},   # not tried again
                       {"errors": [{"message": "server error"}]},
                       {"errors": [{"message": "server error"}]}])
    clips, note = TwitchSource(LINK, run=fake_run(vod_json), fetch=web).viewer_clips()
    assert note == "ok" and [c.offset for c in clips] == [100.0]


def test_a_flaky_page_is_tried_again(vod_json, monkeypatch):
    monkeypatch.setattr(extras, "RETRY_WAIT", 0.0)
    web = FakeWeb(gql=[{"errors": [{"message": "service error"}]},
                       clips_page([clip_node("a", "123", 100, 5)]),
                       clips_page([])])
    clips, note = TwitchSource(LINK, run=fake_run(vod_json), fetch=web).viewer_clips()
    assert note == "ok" and [c.offset for c in clips] == [100.0] and len(web.posts) == 3


def test_viewer_clips_failure_is_missing(vod_json, monkeypatch):
    monkeypatch.setattr(extras, "RETRY_WAIT", 0.0)
    web = FakeWeb(gql=[{"errors": [{"message": "service timeout"}]}] * 4)
    clips, note = TwitchSource(LINK, run=fake_run(vod_json), fetch=web).viewer_clips()
    assert clips is None and note.startswith("missing: ") and "service timeout" in note


@pytest.mark.parametrize("age_days, period", [(0.5, "LAST_DAY"), (3, "LAST_WEEK"),
                                              (20, "LAST_MONTH"), (45, "ALL_TIME")])
def test_clip_period_covers_the_broadcast(age_days, period):
    now = 1_800_000_000.0
    assert extras.clip_period(now - age_days * 86400, now=now) == period


# --- the creator's picture ---------------------------------------------------------------

def png_bytes(mode="RGBA"):
    buf = io.BytesIO()
    Image.new(mode, (300, 300), (200, 30, 30, 128) if mode == "RGBA" else (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


PICTURE = "https://static-cdn.jtvnw.net/jtv_user_pictures/x-profile_image-300x300.png"
CHANNEL_PAGE = (f'<html><head><meta property="og:title" content="Some Streamer"/>'
                f'<meta property="og:image" content="{PICTURE}"/></head></html>').encode()


def test_og_image_is_read_from_the_page():
    assert extras.parse_og_image(CHANNEL_PAGE.decode()) == PICTURE
    assert extras.parse_og_image("<html></html>") is None


def test_avatar_is_saved_as_jpeg(tmp_path, vod_json):
    web = FakeWeb(pages={"https://www.twitch.tv/somestreamer": CHANNEL_PAGE, PICTURE: png_bytes()})
    path, note = TwitchSource(LINK, run=fake_run(vod_json), fetch=web).avatar(tmp_path / "a.jpg")
    assert note == "ok" and path == tmp_path / "a.jpg"
    with Image.open(path) as img:
        assert img.format == "JPEG" and img.size == (300, 300) and img.mode == "RGB"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.jpg"]


def test_avatar_falls_back_to_graphql(tmp_path, vod_json):
    web = FakeWeb(gql=[{"data": {"user": {"profileImageURL": PICTURE}}}],
                  pages={PICTURE: png_bytes("RGB")})
    path, note = TwitchSource(LINK, run=fake_run(vod_json), fetch=web).avatar(tmp_path / "a.jpg")
    assert note == "ok" and path.exists()


def test_avatar_that_is_not_a_picture_is_missing(tmp_path, vod_json):
    web = FakeWeb(pages={"https://www.twitch.tv/somestreamer": CHANNEL_PAGE, PICTURE: b"<html>"})
    path, note = TwitchSource(LINK, run=fake_run(vod_json), fetch=web).avatar(tmp_path / "a.jpg")
    assert path is None and note.startswith("missing: ") and not (tmp_path / "a.jpg").exists()


def test_extras_are_noted_in_the_description(monkeypatch, vod_json):
    monkeypatch.setattr(extras, "chat_tool", lambda: None)
    src = TwitchSource(LINK, run=fake_run(vod_json), fetch=FakeWeb())
    src.chat_times()
    assert src.describe().extras["chat"] == "missing: chat tool not installed"


def test_a_file_has_no_extras(synthetic_stream, tmp_path):
    src = FileSource(synthetic_stream)
    assert src.viewer_clips() == (None, "missing: not a Twitch broadcast")
    assert src.avatar(tmp_path / "a.jpg") == (None, "missing: not a Twitch broadcast")
