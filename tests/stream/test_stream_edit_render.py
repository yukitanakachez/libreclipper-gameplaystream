"""The render graph (panels, cut, text, speed, sounds) and one real encode."""

import re
import wave
from pathlib import Path

import numpy as np
import pytest

from clipper import shell
from streamclip import audio
from streamclip.edit import render
from streamclip.errors import NoVideo, StageError
from streamclip.schema import Layout, Overlay, Placed, Rect, Shot, TextAssets

SPLIT = Shot(0.0, 12.0, "split", Rect(24 / 1280, 24 / 720, 0.25, 0.2), Rect(0.25, 0.0, 0.494, 1.0), Rect(0.05, 0.05, 0.18, 0.14))
# For the 1280x720 test stream: the colour-bar "camera" on top, the middle of the pattern below.
SPLIT_30 = Shot(0.0, 30.0, "split", Rect(24 / 1280, 24 / 720, 0.25, 0.25), Rect(0.25, 0.0, 0.494, 1.0), Rect(0.05, 0.06, 0.18, 0.2))
NO_TEXT = TextAssets(None, [])


def cut(stream: Path, start: float, end: float, folder: Path) -> Path:
    """That stretch of the test stream as its own file, where t=0 is `start`."""
    dest = Path(folder) / f"cut-{start:g}-{end:g}.mp4"
    shell.run([shell.ffmpeg(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
               "-i", str(stream), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", str(dest)])
    return dest


@pytest.fixture
def a_png(tmp_path) -> Path:
    from PIL import Image
    path = tmp_path / "tag.png"
    Image.new("RGBA", (300, 80), (255, 0, 255, 255)).save(path)
    return path


@pytest.fixture
def a_wav(tmp_path) -> Path:
    """Half a second of a loud 1 kHz tone, mono, 44.1 kHz."""
    path = tmp_path / "boom.wav"
    t = np.arange(22050) / 44100
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes((np.sin(2 * np.pi * 1000 * t) * 0.9 * 32767).astype("<i2").tobytes())
    return path


def frame_at(video: Path, t: float) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)


def test_split_builds_two_panels_and_the_seam_line():
    g = render.build_graph([SPLIT], [(0.0, 12.0)], TextAssets(None, []), [], 1.3, 1920, 1080, 30.0, 12.0)
    assert "scale=1080:692" in g and "scale=1080:1228" in g and "vstack=inputs=2" in g
    assert all(int(h) % 2 == 0 for h in re.findall(r"scale=\d+:(\d+)", g))    # 4:2:0 needs even rows
    assert "drawbox" in g and "D7FF3E" in g.upper()


def test_a_game_crop_wider_than_its_panel_is_shown_whole_over_a_blur():
    wide = Shot(0.0, 12.0, "split", SPLIT.face, Rect(0.0, 0.0, 1.0, 1.0), None)
    g = render.build_graph([wide], [(0.0, 12.0)], NO_TEXT, [], 1.3, 1920, 1080, 30.0, 12.0)
    assert "boxblur" in g and "scale=1080:-2" in g and "overlay=(W-w)/2:(H-h)/2" in g
    assert "scale=1080:1228:force_original_aspect_ratio=increase,crop=1080:1228" in g


def test_a_game_crop_that_fits_its_panel_is_not_blurred():
    g = render.build_graph([SPLIT], [(0.0, 12.0)], NO_TEXT, [], 1.3, 1920, 1080, 30.0, 12.0)
    assert "boxblur" not in g


def test_face_and_game_shots_fill_the_frame():
    g = render.build_graph([Shot(0.0, 6.0, "face", Rect(.3, 0, .3164, 1), None, None),
                            Shot(6.0, 12.0, "game", None, Rect(.3418, 0, .3164, 1), None)],
                           [(0.0, 12.0)], TextAssets(None, []), [], 1.3, 1920, 1080, 30.0, 12.0)
    assert g.count("scale=1080:1920") == 2 and "concat=n=2:v=1:a=0" in g and "vstack" not in g


def test_the_zoom_splits_its_shot_in_three():
    shots = render.segments(Layout(1, [SPLIT]), 5.0)
    assert [(s.start, s.end) for s in shots] == [(0.0, 5.0), (5.0, 5.7), (5.7, 12.0)]
    assert shots[1].face == SPLIT.zoom and shots[0].face == SPLIT.face and shots[1].game == SPLIT.game


def test_no_zoom_on_a_game_shot_or_without_a_time():
    game = Shot(0.0, 12.0, "game", None, Rect(.34, 0, .32, 1), None)
    assert len(render.segments(Layout(1, [game]), 5.0)) == 1 and len(render.segments(Layout(1, [SPLIT]), None)) == 1


def test_a_zoom_at_the_edge_of_its_shot_makes_no_empty_piece():
    assert [(s.start, s.end) for s in render.segments(Layout(1, [SPLIT]), 0.0)] == [(0.0, 0.7), (0.7, 12.0)]
    assert [(s.start, s.end) for s in render.segments(Layout(1, [SPLIT]), 11.6)] == [(0.0, 11.6), (11.6, 12.0)]
    assert len(render.segments(Layout(1, [SPLIT]), 12.0)) == 1


def test_only_the_shot_that_holds_the_zoom_is_split():
    later = Shot(12.0, 20.0, "face", Rect(.3, 0, .3164, 1), None, Rect(.35, .1, .2, .6))
    shots = render.segments(Layout(1, [SPLIT, later]), 14.0)
    assert [(s.start, s.end, s.kind) for s in shots] == [(0.0, 12.0, "split"), (12.0, 14.0, "face"),
                                                         (14.0, 14.7, "face"), (14.7, 20.0, "face")]
    assert shots[2].face == later.zoom


def test_order_is_cut_then_text_then_speed():
    text = TextAssets("/w/captions.ass", [Overlay("/w/title.png", 130, 731, 0.0, 3.9), Overlay("/w/tag.png", 40, 600, 0.0, None)])
    g = render.build_graph([SPLIT], [(0.0, 5.0), (7.0, 12.0)], text, [], 1.3, 1920, 1080, 30.0, 12.0)
    assert g.index("select=") < g.index("overlay=130:731") < g.index("ass=") < g.index("setpts=PTS/1.300")
    assert "enable='between(t,0.000,3.900)'" in g and "fontsdir=" in g
    assert "[1:v]overlay=130:731" in g and "[2:v]overlay=40:600[" in g      # the tag is always on


def test_sounds_are_delayed_in_finished_time_and_mixed_under():
    g = render.build_graph([SPLIT], [(0.0, 12.0)], TextAssets(None, []),
                           [Placed("boom", "/s/boom.wav", 6.5, -8.0)], 1.3, 1920, 1080, 30.0, 12.0)
    assert "atempo=1.300" in g and "adelay=5000|5000" in g and "volume=-8.0dB" in g
    assert "amix=inputs=2" in g and "normalize=0" in g and "loudnorm=I=-14:TP=-1:LRA=11" in g


def test_sounds_come_after_the_overlays_in_the_inputs():
    text = TextAssets(None, [Overlay("/w/title.png", 130, 731, 0.0, 3.9), Overlay("/w/tag.png", 40, 600, 0.0, None)])
    g = render.build_graph([SPLIT], [(0.0, 12.0)], text, [Placed("boom", "/s/boom.wav", 6.5, -8.0),
                                                         Placed("pop", "/s/pop.wav", 0.0, -12.0)],
                           1.3, 1920, 1080, 30.0, 12.0)
    assert "[3:a]" in g and "[4:a]" in g and "amix=inputs=3" in g and "adelay=0|0" in g


def test_speed_one_leaves_tempo_alone():
    g = render.build_graph([SPLIT], [(0.0, 12.0)], TextAssets(None, []), [], 1.0, 1920, 1080, 30.0, 12.0)
    assert "atempo" not in g and "setpts=PTS/" not in g


def test_the_measuring_pass_is_sound_only():
    g = render.build_graph([SPLIT], [(0.0, 12.0)], NO_TEXT, [], 1.3, 1920, 1080, 30.0, 12.0, measure=True)
    assert "print_format=json" in g and "[aout]" in g and "[0:v]" not in g and "[vout]" not in g


def test_a_measurement_is_applied_in_the_encode():
    measured = {"input_i": "-23.1", "input_tp": "-4.0", "input_lra": "5.0", "input_thresh": "-33.5",
                "target_offset": "0.3"}
    g = render.build_graph([SPLIT], [(0.0, 12.0)], NO_TEXT, [], 1.3, 1920, 1080, 30.0, 12.0, measured=measured)
    assert "measured_I=-23.1" in g and "linear=true" in g and g.endswith("aresample=48000[aout]")


def test_shots_are_held_inside_the_clip_and_an_empty_layout_is_an_error():
    g = render.build_graph([Shot(-2.0, 20.0, "game", None, Rect(.34, 0, .32, 1), None)],
                           [(0.0, 12.0)], NO_TEXT, [], 1.3, 1920, 1080, 30.0, 12.0)
    assert "trim=start=0.000:end=12.000" in g
    with pytest.raises(StageError):
        render.build_graph([], [(0.0, 12.0)], NO_TEXT, [], 1.3, 1920, 1080, 30.0, 12.0)
    with pytest.raises(StageError):
        render.build_graph([Shot(0.0, 12.0, "face", None, None, None)], [(0.0, 12.0)], NO_TEXT, [],
                           1.3, 1920, 1080, 30.0, 12.0)


def test_a_real_render(synthetic_stream, tmp_path, a_png, a_wav):
    out = render.render_clip(cut(synthetic_stream, 60, 90, tmp_path), Layout(1, [SPLIT_30]),
                             [(0.0, 10.0), (20.0, 30.0)], 5.0,
                             TextAssets(None, [Overlay(str(a_png), 40, 600, 0.0, None)]),
                             [Placed("boom", str(a_wav), 3.0, -8.0)], 1.3, tmp_path / "01-test.mp4")
    info = shell.probe(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert (v["width"], v["height"], v["pix_fmt"]) == (1080, 1920, "yuv420p")
    assert float(v["duration"]) == pytest.approx(20.0 / 1.3, abs=0.1)
    assert abs(float(v["duration"]) - float(a["duration"])) < 0.05
    assert not list(tmp_path.glob("*.partial.mp4"))

    # The picture: the overlay where it was put, the seam line between the panels, and the
    # frozen ten seconds gone (the pattern is still moving just after the cut).
    frame = frame_at(out, 6.0)
    assert np.abs(frame[620:660, 100:300] - (255, 0, 255)).mean() < 40
    assert np.abs(frame[689:693, 500:1000] - (0xD7, 0xFF, 0x3E)).mean() < 40
    game = slice(800, 1800), slice(100, 900)
    assert np.abs(frame_at(out, 8.2)[game] - frame_at(out, 8.5)[game]).mean() > 2.0

    # The sound: the tone lands at 3.0 / 1.3 finished seconds, well over the quiet noise.
    pcm = audio.read_pcm(out)
    at = 3.0 / 1.3
    during = audio.level_db(pcm[int((at + 0.1) * 16000):int((at + 0.4) * 16000)])
    before = audio.level_db(pcm[int((at - 0.5) * 16000):int((at - 0.1) * 16000)])
    assert during > before + 10


def test_a_render_with_captions_and_no_cuts_at_normal_speed(synthetic_stream, tmp_path):
    """Speed 1.0 with nothing cut: the finished clip is as long as its source."""
    ass = tmp_path / "it's [captions].ass"                  # a path that needs escaping
    ass.write_text("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
                   "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, Alignment\n"
                   "Style: Default,Arial,120,&H00FFFFFF,5\n\n"
                   "[Events]\nFormat: Layer, Start, End, Style, Text\n"
                   "Dialogue: 0,0:00:00.00,0:00:04.00,Default,HELLO\n")
    out = render.render_clip(cut(synthetic_stream, 100, 104, tmp_path),
                             Layout(1, [Shot(0.0, 4.0, "game", None, Rect(0.34, 0.0, 0.3164, 1.0), None)]),
                             [(0.0, 4.0)], None, TextAssets(str(ass), []),
                             [Placed("gone", str(tmp_path / "missing.wav"), 1.0, -8.0)], 1.0,
                             tmp_path / "out" / "02-test.mp4")
    v = next(s for s in shell.probe(out)["streams"] if s["codec_type"] == "video")
    assert float(v["duration"]) == pytest.approx(4.0, abs=0.1)


def test_a_source_with_no_picture_is_reported_as_such(synthetic_stream, tmp_path):   # Review Focus 3
    sound_only = tmp_path / "source.mp4"
    shell.run([shell.ffmpeg(), "-v", "error", "-y", "-t", "3", "-i", str(synthetic_stream),
               "-vn", "-c:a", "copy", str(sound_only)])
    with pytest.raises(NoVideo):
        render.render_clip(sound_only, Layout(1, [SPLIT]), [(0.0, 3.0)], None, NO_TEXT, [], 1.3,
                           tmp_path / "x.mp4")
    assert not list(tmp_path.glob("x*"))


def test_a_failed_render_leaves_no_file(tmp_path):
    with pytest.raises(shell.CommandError):
        render.render_clip(tmp_path / "missing.mp4", Layout(1, [SPLIT]), [(0.0, 12.0)], None,
                           TextAssets(None, []), [], 1.3, tmp_path / "x.mp4")
    assert not list(tmp_path.glob("x*"))


def test_a_render_that_fails_halfway_leaves_no_file(synthetic_stream, tmp_path):
    with pytest.raises(shell.CommandError):
        render.render_clip(cut(synthetic_stream, 100, 104, tmp_path), Layout(1, [SPLIT_30]), [(0.0, 4.0)],
                           None, TextAssets(None, [Overlay(str(tmp_path / "missing.png"), 0, 0, 0.0, None)]),
                           [], 1.3, tmp_path / "x.mp4")
    assert not list(tmp_path.glob("x*"))


def test_the_encoder_uses_the_processors_this_job_may_use(monkeypatch):
    from streamclip import settings
    from streamclip.edit import render
    monkeypatch.setattr(settings, "processors", lambda: 4)
    options = render.encode_options()
    assert options[options.index("-threads") + 1] == "4"
    assert options[options.index("-preset") + 1] == "fast"


def test_the_preset_can_be_set_from_outside(monkeypatch):
    from streamclip.edit import render

    def preset():
        options = render.encode_options()
        return options[options.index("-preset") + 1]
    monkeypatch.delenv("STREAMCLIP_X264_PRESET", raising=False)
    assert preset() == "fast"
    monkeypatch.setenv("STREAMCLIP_X264_PRESET", "ultrafast")       # what Starwish sets
    assert preset() == "ultrafast"
    monkeypatch.setenv("STREAMCLIP_X264_PRESET", "  ")
    assert preset() == "fast"
