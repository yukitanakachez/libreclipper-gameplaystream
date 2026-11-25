"""A Twitch past broadcast as a stream source. No test here touches the network: the
downloader is replaced by a fake runner."""

import json
from pathlib import Path

import pytest

from clipper import shell
from streamclip.errors import Refused, StageError
from streamclip.source import open_source, twitch
from streamclip.source.twitch import TwitchSource

FIXTURES = Path(__file__).parent / "fixtures" / "twitch"
LINK = "https://www.twitch.tv/videos/123"


@pytest.fixture
def vod_json() -> str:
    return (FIXTURES / "vod.json").read_text()


def fake_run(answer=None, fail=None):
    def run(cmd, timeout=3600):
        if fail:
            raise shell.CommandError(cmd, 1, fail)
        return answer
    return run


def no_network(cmd, timeout=3600):
    raise AssertionError(f"the runner was called: {cmd}")


def test_describe_reads_the_downloader_json(vod_json):
    info = TwitchSource("https://www.twitch.tv/videos/123", run=fake_run(vod_json)).describe()
    assert info.kind == "twitch" and info.video_id == "123"
    assert (info.creator_login, info.creator_name) == ("somestreamer", "Some Streamer")
    assert info.duration > 0 and info.height <= 1080 and info.chapters and info.chapters[0].title
    assert info.source == LINK and info.title.startswith("[683/730]")


def test_describe_details(vod_json):
    info = TwitchSource(LINK, run=fake_run(vod_json)).describe()
    assert (info.width, info.height, round(info.fps)) == (1920, 1080, 60)
    assert info.started_at.startswith("2026-") and info.started_at.endswith("+00:00")
    assert [c.title for c in info.chapters][:2] == ["Just Chatting", "Chess"]
    assert info.chapters[1].start == 4451 and info.chapters[-1].end == info.duration
    assert set(info.extras) == {"chat", "viewer_clips", "avatar"}


def test_describe_runs_the_downloader_as_a_module(vod_json):
    calls = []

    def run(cmd, timeout=3600):
        calls.append(cmd)
        return vod_json
    TwitchSource(LINK, run=run).describe()
    cmd = calls[0]
    assert cmd[1:3] == ["-m", "yt_dlp"] and "-J" in cmd and cmd[-1] == LINK


def test_best_format_is_at_most_1080(vod_json):
    answer = json.loads(vod_json)
    answer["formats"].append({"format_id": "chunked", "width": 2560, "height": 1440, "fps": 60.0,
                              "vcodec": "avc1", "acodec": "mp4a"})
    info = TwitchSource(LINK, run=fake_run(json.dumps(answer))).describe()
    assert info.height == 1080


@pytest.mark.parametrize("link", [
    "https://www.twitch.tv/videos/123", "https://twitch.tv/videos/123?t=1h2m3s",
    "https://m.twitch.tv/videos/123", "https://www.twitch.tv/somestreamer/v/123",
    "twitch.tv/videos/123"])
def test_broadcast_links_are_accepted(link, vod_json):
    src = open_source(link)
    assert isinstance(src, TwitchSource) and src.video_id == "123"


@pytest.mark.parametrize("link, reason", [
    ("https://www.twitch.tv/somestreamer", "channel, not a past broadcast"),
    ("https://www.twitch.tv/somestreamer/videos", "channel, not a past broadcast"),
    ("https://clips.twitch.tv/SomeClipName", "single clip"),
    ("https://www.twitch.tv/somestreamer/clip/SomeClipName", "single clip")])
def test_link_shape_is_refused_without_the_network(link, reason):
    with pytest.raises(Refused, match=reason):
        TwitchSource(link, run=no_network).describe()
    with pytest.raises(Refused, match=reason):
        open_source(link).describe()          # must not call the runner


def test_channel_refusal_is_the_whole_sentence():
    with pytest.raises(Refused) as caught:
        open_source("https://www.twitch.tv/somestreamer").describe()
    assert str(caught.value) == ("That link is a channel, not a past broadcast. Send the link of "
                                 "one finished broadcast, like twitch.tv/videos/123456.")


@pytest.mark.parametrize("stderr, reason", [
    ("ERROR: [twitch:vod] 123: This video is only available to subscribers",
     "That broadcast is for subscribers only."),
    ("ERROR: [twitch:vod] 123: You must be logged into an account that has access to this "
     "subscriber-only content. Use --cookies-from-browser", "That broadcast is for subscribers only."),
    ("ERROR: [twitch:vod] 123: Video 123 does not exist",
     "That broadcast no longer exists. Twitch deletes past broadcasts after 7 to 60 days.")])
def test_downloader_errors_become_plain_reasons(stderr, reason):
    with pytest.raises(Refused) as caught:
        TwitchSource(LINK, run=fake_run(fail=stderr)).describe()
    assert str(caught.value) == reason


def test_other_downloader_failures_are_stage_errors():
    with pytest.raises(StageError, match="Unable to download JSON metadata") as caught:
        TwitchSource(LINK, run=fake_run(fail="ERROR: [twitch:vod] 123: Unable to download "
                                             "JSON metadata: HTTP Error 500")).describe()
    assert not isinstance(caught.value, Refused)


def test_live_and_overlong_are_refused(vod_json):
    live = {**json.loads(vod_json), "live_status": "is_live", "duration": None}
    with pytest.raises(Refused) as caught:
        TwitchSource(LINK, run=fake_run(json.dumps(live))).describe()
    assert str(caught.value) == "That broadcast is still live. It can be clipped once it has ended."

    long = {**json.loads(vod_json), "duration": 7 * 3600}
    with pytest.raises(Refused) as caught:
        TwitchSource(LINK, run=fake_run(json.dumps(long))).describe()
    assert str(caught.value) == "That broadcast is over 6 hours long. The limit is 6 hours."


def test_sound_only_broadcast_is_refused(vod_json):
    answer = json.loads(vod_json)
    answer["formats"] = [f for f in answer["formats"] if f["vcodec"] in (None, "none")]
    with pytest.raises(Refused, match="no video"):
        TwitchSource(LINK, run=fake_run(json.dumps(answer))).describe()


# --- Fetching sound and video stretches ------------------------------------------------

from streamclip import audio  # noqa: E402
from streamclip.errors import NoVideo  # noqa: E402


class Recorder:
    """A fake downloader: each yt-dlp call is recorded and answered by copying a stretch of
    the synthetic stream to the path after -o. ffmpeg and ffprobe calls run for real.
    A section *a-b inside the stream's 2 minutes is answered with that very section; one
    beyond it with the stream from a % 20 s, at most 100 s long."""

    def __init__(self, stream, sound_only=False):
        self.stream, self.sound_only, self.calls = stream, sound_only, []

    def __call__(self, cmd, timeout=3600):
        cmd = [str(c) for c in cmd]
        if cmd[1:3] != ["-m", "yt_dlp"]:
            return shell.run(cmd, timeout=timeout)
        self.calls.append(cmd)
        out = cmd[cmd.index("-o") + 1]
        if "--download-sections" in cmd:
            a, b = map(float, cmd[cmd.index("--download-sections") + 1].lstrip("*").split("-"))
            first, length = (a, b - a) if b <= 120 else (a % 20, min(b - a, 100.0))
            span = ["-ss", f"{first}", "-t", f"{length}"]
        else:
            span = []
        flags = ["-vn", "-c:a", "copy"] if (self.sound_only or "Audio_Only" in " ".join(cmd)) else ["-c", "copy"]
        shell.run([shell.ffmpeg(), "-v", "error", "-y", *span, "-i", str(self.stream), *flags, out])
        return ""


@pytest.fixture
def recorder(synthetic_stream):
    return Recorder(synthetic_stream)


@pytest.fixture
def sound_only_runner(synthetic_stream):
    return Recorder(synthetic_stream, sound_only=True)


def test_audio_command_asks_for_sound_only(tmp_path, recorder):
    out = TwitchSource(LINK, run=recorder).fetch_audio(tmp_path / "audio.m4a")
    cmd = recorder.calls[0]
    assert cmd[1:3] == ["-m", "yt_dlp"] and "Audio_Only/bestaudio" in cmd[cmd.index("-f") + 1]
    assert cmd[-1] == LINK and "--ffmpeg-location" in cmd
    streams = shell.probe(out)["streams"]
    assert [s["codec_type"] for s in streams] == ["audio"] and streams[0]["channels"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audio.m4a"]       # nothing left over


def test_video_command_asks_for_a_padded_section(tmp_path, recorder):
    TwitchSource(LINK, run=recorder).fetch_video(600.0, 690.0, 360, tmp_path / "w.mp4")
    cmd = recorder.calls[0]
    assert cmd[cmd.index("--download-sections") + 1] == "*595.0-695.0"
    assert "height<=360" in cmd[cmd.index("-f") + 1]


def test_section_at_the_start_does_not_go_negative(tmp_path, recorder):
    TwitchSource(LINK, run=recorder).fetch_video(2.0, 92.0, 360, tmp_path / "w.mp4")
    assert recorder.calls[0][recorder.calls[0].index("--download-sections") + 1].startswith("*0.0-")


def test_fetched_stretch_is_trimmed_exactly(tmp_path, recorder):
    # The piece is *25.0-40.0 of the synthetic stream; its 5 s of padding are cut away, so
    # the stretch asked for, 30-35, begins with the burst at 30 s.
    out = TwitchSource(LINK, run=recorder).fetch_video(30.0, 35.0, 360, tmp_path / "w.mp4")
    info = shell.probe(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video["height"] == 360 and abs(float(info["format"]["duration"]) - 5.0) < 0.1
    levels = audio.levels_db(audio.read_pcm(out, 0, 3.0), 16000, 0.01)
    onset = next(i for i, v in enumerate(levels) if v > -15) * 0.01
    assert onset < 0.05
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.mp4"]


def test_sound_only_answer_raises_no_video(tmp_path, sound_only_runner):     # Review Focus 3
    with pytest.raises(NoVideo, match="sound only"):
        TwitchSource(LINK, run=sound_only_runner).fetch_video(600.0, 690.0, 1080, tmp_path / "s.mp4")
    assert not (tmp_path / "s.mp4").exists()


def test_download_failure_is_a_stage_error(tmp_path):
    with pytest.raises(StageError):
        TwitchSource(LINK, run=fake_run(fail="ERROR: unable to download video data: HTTP Error 403")
                     ).fetch_video(600.0, 690.0, 360, tmp_path / "w.mp4")
    assert list(tmp_path.iterdir()) == []


def test_fractional_start_still_lines_up(tmp_path, recorder):
    # The section sent to the downloader and the offset trimmed by must agree; rounding
    # one and not the other shifts the stretch by up to 0.05 s.
    out = TwitchSource(LINK, run=recorder).fetch_video(29.96, 34.0, 360, tmp_path / "w.mp4")
    sent = recorder.calls[0][recorder.calls[0].index("--download-sections") + 1]
    assert sent == "*24.9-39.0"
    levels = audio.levels_db(audio.read_pcm(out, 0, 1.0), 16000, 0.005)
    onset = next(i for i, v in enumerate(levels) if v > -15) * 0.005
    assert abs(onset - 0.04) < 0.02


@pytest.mark.parametrize("start, end", [(-3.0, 10.0), (10.0, 10.0), (10.0, 5.0)])
def test_a_stretch_must_start_at_zero_or_later_and_have_length(tmp_path, recorder, start, end):
    with pytest.raises(ValueError):
        TwitchSource(LINK, run=recorder).fetch_video(start, end, 360, tmp_path / "w.mp4")
    assert recorder.calls == []


TIMEOUT = ("ERROR: [twitch:vod] 123: Failed to download m3u8 information: "
           "HTTPSConnectionPool(host='usher.ttvnw.net', port=443): Read timed out. "
           "(read timeout=30.0) (caused by TransportError(\"...Read timed out.\"))")


def flaky_run(answers):
    """A runner that fails with each stderr in `answers` until it reaches a plain answer."""
    calls = []

    def run(cmd, timeout=3600):
        calls.append(cmd)
        answer = answers[min(len(calls), len(answers)) - 1]
        if answer.startswith("ERROR:"):
            raise shell.CommandError(cmd, 1, answer)
        return answer
    run.calls = calls
    return run


def test_a_network_timeout_is_tried_again(vod_json, monkeypatch):
    monkeypatch.setattr(twitch, "RETRY_WAIT", 0.0)
    run = flaky_run([TIMEOUT, TIMEOUT, vod_json])
    info = TwitchSource(LINK, run=run).describe()
    assert info.video_id == "123" and len(run.calls) == 3


def test_a_timeout_that_keeps_happening_is_a_stage_error(monkeypatch):
    monkeypatch.setattr(twitch, "RETRY_WAIT", 0.0)
    run = flaky_run([TIMEOUT])
    with pytest.raises(StageError, match="Read timed out"):
        TwitchSource(LINK, run=run).describe()
    assert len(run.calls) == twitch.NETWORK_TRIES


def test_a_refusal_is_not_tried_again(monkeypatch):
    monkeypatch.setattr(twitch, "RETRY_WAIT", 0.0)
    run = flaky_run(["ERROR: [twitch:vod] 123: This video is only available to subscribers"])
    with pytest.raises(Refused):
        TwitchSource(LINK, run=run).describe()
    assert len(run.calls) == 1
