"""A video file on disk as a stream source."""

import subprocess
from pathlib import Path

import pytest

from clipper import shell
from streamclip import audio
from streamclip.errors import Refused
from streamclip.source import open_source
from streamclip.source.local import FileSource


def test_describe(synthetic_stream):
    info = open_source(str(synthetic_stream)).describe()
    assert info.kind == "file" and abs(info.duration - 120) < 0.5
    assert (info.width, info.height, round(info.fps)) == (1280, 720, 30)
    assert info.creator_name == "" and info.title == synthetic_stream.stem
    assert set(info.extras) == {"chat", "viewer_clips", "avatar"}
    assert all(v == "missing: not a Twitch broadcast" for v in info.extras.values())


def test_open_source_gives_a_file_source(synthetic_stream):
    assert isinstance(open_source(str(synthetic_stream)), FileSource)


def test_fetch_audio_is_whole_and_sound_only(synthetic_stream, tmp_path):
    out = open_source(str(synthetic_stream)).fetch_audio(tmp_path / "audio.m4a")
    streams = shell.probe(out)["streams"]
    assert [s["codec_type"] for s in streams] == ["audio"]
    assert streams[0]["channels"] == 1
    assert abs(float(shell.probe(out)["format"]["duration"]) - 120) < 0.5


def test_fetch_video_is_exact_and_scaled(synthetic_stream, tmp_path):
    out = open_source(str(synthetic_stream)).fetch_video(29.0, 34.0, 360, tmp_path / "w.mp4")
    video = next(s for s in shell.probe(out)["streams"] if s["codec_type"] == "video")
    assert video["height"] == 360 and video["width"] == 640
    assert abs(float(shell.probe(out)["format"]["duration"]) - 5.0) < 0.1
    # the loud burst at 30-32 s of the stream sits at 1-3 s of the stretch
    assert audio.level_db(audio.read_pcm(out, 1.2, 1.5)) > audio.level_db(audio.read_pcm(out, 3.5, 1.0)) + 10


def test_fetch_video_starts_exactly_on_the_start(synthetic_stream, tmp_path):
    # The burst starts at 30.0 s of the stream, so at 2.0 s of a stretch from 28.0 s.
    out = open_source(str(synthetic_stream)).fetch_video(28.0, 34.0, 360, tmp_path / "w.mp4")
    pcm = audio.read_pcm(out, 0, 4.0)
    levels = audio.levels_db(pcm, 16000, 0.01)
    onset = next(i for i, v in enumerate(levels) if v > -15) * 0.01
    assert abs(onset - 2.0) < 0.05


def test_fetch_video_never_scales_up(synthetic_stream, tmp_path):
    out = open_source(str(synthetic_stream)).fetch_video(10.0, 12.0, 1080, tmp_path / "w.mp4")
    video = next(s for s in shell.probe(out)["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1280, 720)


def test_extras_are_missing_with_a_reason(synthetic_stream, tmp_path):
    src = open_source(str(synthetic_stream))
    assert src.chat_times() == (None, "missing: not a Twitch broadcast")
    assert src.viewer_clips() == (None, "missing: not a Twitch broadcast")
    assert src.avatar(tmp_path / "a.jpg")[0] is None


def _ffmpeg_copy(src, dest, *flags):
    subprocess.run([shell.ffmpeg(), "-v", "error", "-y", "-t", "3", "-i", str(src), *flags,
                    "-c", "copy", str(dest)], check=True, capture_output=True)


@pytest.mark.parametrize("make, reason", [
    ("no_audio", "no sound"), ("no_video", "no video"), ("missing", "No file at")])
def test_refusals(tmp_path, make, reason, synthetic_stream):
    if make == "no_audio":
        path = tmp_path / "silent.mp4"
        _ffmpeg_copy(synthetic_stream, path, "-an")
    elif make == "no_video":
        path = tmp_path / "sound.m4a"
        _ffmpeg_copy(synthetic_stream, path, "-vn")
    else:
        path = tmp_path / "nothing.mp4"
    with pytest.raises(Refused, match=reason):
        open_source(str(path)).describe()


def test_not_a_video_is_refused(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video")
    with pytest.raises(Refused):
        open_source(str(path)).describe()


def test_overlong_file_is_refused(synthetic_stream):
    def fake_run(cmd, timeout=3600):
        return ('{"format": {"duration": "25200"}, "streams": ['
                '{"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "30/1"},'
                '{"codec_type": "audio"}]}')
    with pytest.raises(Refused, match="That recording is over 6 hours long. The limit is 6 hours."):
        FileSource(str(synthetic_stream), run=fake_run).describe()


def test_other_sites_are_refused():
    with pytest.raises(Refused, match="Only Twitch links and video files"):
        open_source("https://www.youtube.com/watch?v=abc")


@pytest.mark.parametrize("start, end", [(-3.0, 10.0), (10.0, 10.0), (10.0, 5.0)])
def test_a_stretch_must_start_at_zero_or_later_and_have_length(synthetic_stream, tmp_path, start, end):
    with pytest.raises(ValueError):
        open_source(str(synthetic_stream)).fetch_video(start, end, 360, tmp_path / "w.mp4")


def test_a_cut_stretch_uses_the_processors_this_job_may_use(monkeypatch, tmp_path):
    from streamclip.source import media
    monkeypatch.setattr(media, "processors", lambda: 4)
    seen = []

    def run(cmd, timeout=3600):
        seen.append(cmd)
        if "-show_streams" in cmd:
            return '{"streams": [{"codec_type": "video"}], "format": {}}'
        Path(cmd[-1]).write_bytes(b"x")
        return ""
    media.cut_stretch(tmp_path / "in.mp4", 0.0, 5.0, 360, tmp_path / "out.mp4", run)
    assert seen[0][seen[0].index("-threads") + 1] == "4"
