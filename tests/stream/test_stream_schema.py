import dataclasses
import json

from clipper.models import Rect
from streamclip import schema as m


def roundtrip(obj):
    return type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))


def samples():
    pick = m.Pick(1, 7, 100.0, 140.0, 9, "why", "Title Here Now", "cap", ["#a"],
                  [m.Beat("payoff", 130.0)], [(110.0, 112.5)], False)
    layout = m.Layout(1, [m.Shot(0.0, 40.0, "split", Rect(0, 0, .25, .25), Rect(.25, 0, .5, 1), None)])
    info = m.StreamInfo("https://www.twitch.tv/videos/1", "twitch", "1", "T", "login", "Name",
                        3600.0, "", 1920, 1080, 60.0, [m.Chapter(0.0, 3600.0, "Game")], {"chat": "ok"})
    hot = m.Hotspot(7, 160.0, 100.0, 190.0, 0.9, [m.Fired("voice", 161.0, 0.8, "Screaming")])
    sig = m.Signals(2.0, [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], None, None, [0.0, 0.0], [0.0, 1.0])
    return (pick, layout, info, hot, sig, m.CameraBox(Rect(0, 0, .25, .25), 0.9),
            m.TextAssets("a.ass", [m.Overlay("t.png", 130, 731, 0.0, 3.9)]),
            m.Placed("boom", "boom.wav", 12.0, -10.0), m.MakeOptions(),
            m.CandidateIndex("en", True, [1, 2], []),
            m.Signals(1.0, [0.5], [0.5], [0.5], [1.0], [0.0], [0.0], [0.5]),
            m.Shot(0.0, 2.0, "face", Rect(0, 0, 1, 1), None, Rect(.25, .25, .5, .5)),
            m.TextAssets(None, [m.Overlay("tag.png", 40, 600, 0.0, None)]),
            m.Overlay("title.png", 130, 731, 0.0, 3.9),
            m.ViewerClip(1234.5, 30.0, 12000, "no way"),
            m.Sound("boom", "boom.wav", "payoff", -10.0, 0.05),
            m.Chapter(0.0, 10.0, "Just Chatting"), m.Beat("hard_cut", 3.0),
            m.Fired("chat", 62.0, 0.7, "chat ×4"),
            m.ClipResult(1, "01-title.mp4", "01-title.txt", "Title", 9, "why", 100.0, 140.0, 28.4,
                         False, ["split", "face"], 3, 4.5, ["boom"], True))


def test_every_dataclass_survives_json():
    for obj in samples():
        assert roundtrip(obj) == obj


def test_no_dataclass_is_left_untested():
    declared = {c for c in vars(m).values()
                if dataclasses.is_dataclass(c) and c.__module__ == m.__name__}
    assert declared == {type(obj) for obj in samples()}


def test_protect_comes_back_as_tuples():
    pick = m.Pick(1, 7, 0.0, 30.0, 5, "", "A B C", "", [], [], [(1.0, 2.0)], True)
    assert roundtrip(pick).protect == [(1.0, 2.0)]


def test_the_podcast_types_are_reused_not_copied():
    from clipper import models
    assert (m.Rect, m.Word, m.Transcript) == (models.Rect, models.Word, models.Transcript)


def test_make_options_defaults():
    assert m.MakeOptions() == m.MakeOptions(speed=1.3, dead_air=0.5, sounds=True)
