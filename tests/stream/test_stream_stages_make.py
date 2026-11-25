"""The make stage: picks become finished clips, a caption file each, and the list of them."""

import pytest

from clipper import shell
from clipper.jobs import read_json, write_json
from streamclip import stages
from streamclip.errors import NoVideo, StageError
from streamclip.jobdir import JobDir
from streamclip.schema import MakeOptions
from test_stream_stages_support import (  # noqa: F401  (fixtures)
    PICK1_CAPTION, SECOND_LISTEN, fakes, judged_job, made_job,
)

SOUND_ONLY = "The video for that stretch came back as sound only."


def listing_of(job):
    return read_json(job.out / "clips.json")


def make_fetch_fail_for_rank(monkeypatch, job, rank, exc):
    """The source's video fetch raises `exc` for one rank's stretch, or for every stretch."""
    starts = [p["start"] for p in read_json(job.picks)["picks"] if rank is None or p["rank"] == rank]
    real = stages.open_source

    def opened(source):
        src = real(source)
        fetch = src.fetch_video

        def fetch_video(start, end, max_height, dest):
            if any(abs(start - s) < 0.01 for s in starts):
                raise exc
            return fetch(start, end, max_height, dest)
        src.fetch_video = fetch_video
        return src
    monkeypatch.setattr(stages, "open_source", opened)


def test_make_writes_clips_captions_and_the_list(made_job):
    listing = listing_of(made_job)
    assert [c["rank"] for c in listing["clips"]] == [1, 2] and listing["captions"] is True
    assert listing["notes"] == []
    for c in listing["clips"]:
        video = made_job.out / c["file"]
        streams = shell.probe(video)["streams"]
        v = next(s for s in streams if s["codec_type"] == "video")
        assert (v["width"], v["height"]) == (1080, 1920) and (made_job.out / c["caption_file"]).exists()
        assert any(s["codec_type"] == "audio" for s in streams)
        assert c["file"].startswith(f"{c['rank']:02d}-") and c["layout"] == ["split"]
        assert c["duration"] == pytest.approx(float(shell.probe(video)["format"]["duration"]), abs=0.1)
        assert 15.0 <= c["duration"] <= 40.0
        assert c["captions"] is True and c["weak"] is False and c["sounds"]
        assert c["source_end"] - c["source_start"] == pytest.approx(c["duration"] * 1.3 + c["seconds_cut"], abs=0.5)
    assert listing["clips"][0]["file"] == "01-the-first-loud-one.mp4"
    assert read_json(made_job.camera)["camera"] is not None
    assert made_job.get_status()["stage"] == "done"


def test_every_step_of_a_clip_is_kept_in_its_folder(made_job):
    for rank in (1, 2):
        folder = made_job.clip(rank)
        for name in ("source.mp4", "layout.json", "cut.json", "captions.ass", "title.png"):
            assert (folder / name).exists(), name
        cut = read_json(folder / "cut.json")
        assert cut["keep"] and cut["fps"] == pytest.approx(30.0)


def test_the_caption_file_is_ready_to_paste(made_job):
    text = (made_job.out / listing_of(made_job)["clips"][0]["caption_file"]).read_text()
    assert text.splitlines()[0] == PICK1_CAPTION and text.splitlines()[-1] == "#gaming #funny"
    assert text.splitlines()[1] == ""


def test_a_pick_with_no_caption_gets_its_title_to_paste(made_job):
    picks = read_json(made_job.picks)
    picks["picks"][0].update(caption="", hashtags=[])
    write_json(made_job.picks, picks)
    stages.make(made_job, only=1)
    text = (made_job.out / listing_of(made_job)["clips"][0]["caption_file"]).read_text()
    assert text.strip() == "The first loud one"


@pytest.mark.parametrize("title", ["", "  ", "stream title"])
def test_a_pick_with_no_real_title_gets_no_sticker_and_a_plain_name(made_job, title):
    # A pick made without the judge has no title, and the stream's own name (for a file, the
    # file's name) is not a title either: no sticker is drawn and the file is NN-clip.mp4.
    if title == "stream title":
        title = read_json(made_job.stream)["title"]
    picks = read_json(made_job.picks)
    picks["picks"][0].update(title=title)
    write_json(made_job.picks, picks)
    lines = []
    stages.make(made_job, only=1, progress=lines.append)
    clip = listing_of(made_job)["clips"][0]
    assert clip["file"] == "01-clip.mp4" and clip["title"] == ""
    done = next(line for line in lines if line.startswith("Clip 01 done"))
    assert "“" not in done and done.startswith("Clip 01 done, ")
    assert (made_job.out / "01-clip.mp4").exists() and not (made_job.out / "01-the-first-loud-one.mp4").exists()
    assert not (made_job.clip(1) / "title.png").exists()


def test_a_sound_only_stretch_skips_that_clip_and_keeps_the_rest(judged_job, monkeypatch):   # Review Focus 3
    make_fetch_fail_for_rank(monkeypatch, judged_job, 2, NoVideo(SOUND_ONLY))
    lines = []
    stages.make(judged_job, progress=lines.append)
    listing = listing_of(judged_job)
    assert [c["rank"] for c in listing["clips"]] == [1]
    assert listing["notes"] == [f"Clip 02 was skipped: {SOUND_ONLY}"]
    assert f"Clip 02 was skipped: {SOUND_ONLY}" in lines
    assert judged_job.get_status()["stage"] == "done"


def test_no_clip_at_all_is_a_failure(judged_job, monkeypatch):
    make_fetch_fail_for_rank(monkeypatch, judged_job, None, NoVideo("sound only"))
    with pytest.raises(StageError, match="No clip could be made: sound only"):
        stages.make(judged_job)
    assert listing_of(judged_job)["clips"] == []


def test_only_remakes_one_clip_and_keeps_the_others_listed(made_job):
    before = (made_job.out / listing_of(made_job)["clips"][0]["file"]).stat().st_mtime_ns
    stages.make(made_job, only=2, options=MakeOptions(speed=1.0, dead_air=5.0, sounds=False))
    listing = listing_of(made_job)
    assert len(listing["clips"]) == 2 and listing["clips"][1]["sounds"] == []
    assert listing["clips"][1]["duration"] > 20        # played at its own speed this time
    assert (made_job.out / listing["clips"][0]["file"]).stat().st_mtime_ns == before


def test_only_for_a_clip_that_is_not_there_says_so(made_job):
    with pytest.raises(StageError, match="There is no clip 7"):
        stages.make(made_job, only=7)


def test_a_pick_that_moved_is_fetched_again_and_its_old_files_go(made_job):
    old = listing_of(made_job)["clips"][1]
    picks = read_json(made_job.picks)
    picks["picks"][1]["start"] += 1.0
    picks["picks"][1]["title"] = "A new name"
    write_json(made_job.picks, picks)
    stages.make(made_job, only=2)
    new = listing_of(made_job)["clips"][1]
    assert new["file"] == "02-a-new-name.mp4" and new["source_start"] == picks["picks"][1]["start"]
    assert new["duration"] < old["duration"]
    assert not (made_job.out / old["file"]).exists() and not (made_job.out / old["caption_file"]).exists()


def test_another_output_folder(made_job, tmp_path):
    stages.make(made_job, out=tmp_path / "elsewhere", only=1)
    assert [c["rank"] for c in read_json(tmp_path / "elsewhere" / "clips.json")["clips"]] == [1]
    assert list((tmp_path / "elsewhere").glob("01-*.mp4"))


def test_captions_off_is_said_in_the_list(made_job):
    index = read_json(made_job.candidates_index)
    write_json(made_job.candidates_index, {**index, "captions": False})
    stages.make(made_job, only=1)
    listing = listing_of(made_job)
    assert listing["captions"] is False and "no captions" in listing["notes"][0].lower()
    assert listing["clips"][0]["captions"] is False
    assert not (made_job.clip(1) / "captions.ass").exists()


def test_a_camera_that_cannot_be_looked_for_does_not_stop_the_clips(judged_job, monkeypatch):
    def broken(windows, detect=None):
        raise RuntimeError("the face model would not load")
    monkeypatch.setattr("streamclip.framing.find_camera", broken)
    lines = []
    stages.make(judged_job, only=1, progress=lines.append)
    assert listing_of(judged_job)["clips"] and not judged_job.camera.exists()
    assert any("the face model would not load" in line for line in lines)


def test_progress_names_each_clip(judged_job, monkeypatch):
    make_fetch_fail_for_rank(monkeypatch, judged_job, None, NoVideo("sound only"))
    lines = []
    with pytest.raises(StageError):
        stages.make(judged_job, progress=lines.append)
    assert any(line.startswith("Clip 1 of 2:") for line in lines)
    assert any(line.startswith("Clip 2 of 2:") for line in lines)


def test_making_before_judging_is_a_plain_failure(tmp_path):
    with pytest.raises(StageError, match="Run find and judge first"):
        stages.make(JobDir(tmp_path / "empty"))


def test_a_remake_that_fails_keeps_the_clip_that_was_there(made_job, monkeypatch):
    before = listing_of(made_job)["clips"]

    def broken(*args, **kwargs):
        raise StageError("the encoder fell over")
    monkeypatch.setattr("streamclip.edit.render_clip", broken)
    with pytest.raises(StageError, match="No clip could be made: the encoder fell over"):
        stages.make(made_job, only=2)
    listing = listing_of(made_job)
    assert listing["clips"] == before and (made_job.out / before[1]["file"]).exists()
    assert listing["notes"] == ["Clip 02 was skipped: the encoder fell over"]


def test_another_jobs_clips_in_the_same_folder_are_left_alone(made_job, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "01-theirs.mp4").write_bytes(b"x")
    (shared / "03-theirs.mp4").write_bytes(b"x")
    theirs = {**listing_of(made_job)["clips"][0], "file": "01-theirs.mp4", "caption_file": "01-theirs.txt"}
    write_json(shared / "clips.json", {"source": "another stream", "captions": True, "notes": [],
                                       "clips": [theirs, {**theirs, "rank": 3, "file": "03-theirs.mp4"}]})
    stages.make(made_job, out=shared, only=1)
    listing = read_json(shared / "clips.json")
    assert [c["file"] for c in listing["clips"]] == ["01-the-first-loud-one.mp4"]
    assert listing["source"] == read_json(made_job.request)["source"]
    assert (shared / "01-theirs.mp4").exists() and (shared / "03-theirs.mp4").exists()


def words_of(job, rank):
    return read_json(job.clip(rank) / "words.json")


def test_each_clip_is_heard_again_and_those_words_make_the_captions(made_job):
    for clip in listing_of(made_job)["clips"]:
        heard = words_of(made_job, clip["rank"])
        assert heard["pass"] == "second" and {w["text"] for w in heard["words"]} <= set(SECOND_LISTEN)
        captions = (made_job.clip(clip["rank"]) / "captions.ass").read_text().upper()
        assert "YES" in captions and "LIKE" not in captions
    picks = read_json(made_job.picks)["picks"]
    assert [c["source_start"] for c in listing_of(made_job)["clips"]] == [p["start"] for p in picks]
    assert listing_of(made_job)["notes"] == []


def test_a_second_listen_that_fails_falls_back_to_the_first_and_says_so(made_job, monkeypatch):
    def broken():
        raise RuntimeError("the model would not load")
    monkeypatch.setattr("streamclip.windows.transcribe.caption_model", broken)
    stages.make(made_job, only=1)
    assert words_of(made_job, 1)["pass"] == "first"
    assert "LIKE" in (made_job.clip(1) / "captions.ass").read_text().upper()
    assert listing_of(made_job)["notes"] == [
        "Clip 01 uses the first listen's words: the second listen did not work (the model would not load)."]
    assert words_of(made_job, 2)["pass"] == "second"            # the other clip is as it was


def test_a_second_listen_that_hears_nothing_falls_back_too(made_job, monkeypatch):
    from test_stream_stages_support import FakeSpeech
    silent = FakeSpeech(vocabulary=())
    silent.transcribe = lambda pcm, **how: (iter([]), None)
    monkeypatch.setattr("streamclip.windows.transcribe.caption_model", lambda: silent)
    lines = []
    stages.make(made_job, only=2, progress=lines.append)
    assert words_of(made_job, 2)["pass"] == "first"
    assert listing_of(made_job)["notes"] == [
        "Clip 02 uses the first listen's words: the second listen did not work (it heard no words)."]
    assert any(line.startswith("Clip 02: second listen, 0 words in") for line in lines)


def test_a_slow_render_keeps_saying_it_is_working(made_job, monkeypatch):
    # Starwish gives up on a step that prints nothing for 300 seconds.
    import time

    from streamclip import edit
    real = edit.render_clip
    stamps = []

    def slow(*args, **kwargs):
        stamps.append(made_job.get_status()["updated"])
        time.sleep(0.6)
        stamps.append(made_job.get_status()["updated"])
        return real(*args, **kwargs)
    monkeypatch.setattr("streamclip.edit.render_clip", slow)
    monkeypatch.setattr(stages, "QUIET_AFTER", 0.1)
    lines = []
    stages.make(made_job, only=1, progress=lines.append)
    assert any(line.startswith("Still working: clip 1 of 2 · rendering") for line in lines)
    assert stamps[0] != stamps[1]                    # the status file is touched too
    assert lines[-1].startswith("Done:")             # and nothing is said after the last line


def test_a_progress_that_fails_once_does_not_silence_the_stage(made_job, monkeypatch):
    import time

    from streamclip import edit
    real = edit.render_clip
    monkeypatch.setattr("streamclip.edit.render_clip", lambda *a, **k: time.sleep(0.8) or real(*a, **k))
    monkeypatch.setattr(stages, "QUIET_AFTER", 0.1)
    lines = []

    def progress(line):
        if line.startswith("Still working") and not any(l.startswith("Still") for l in lines):
            lines.append(line)
            raise OSError("the pipe hiccupped")
        lines.append(line)
    stages.make(made_job, only=1, progress=progress)
    assert sum(1 for line in lines if line.startswith("Still working")) >= 2


def test_each_clip_is_framed_from_its_own_full_size_stretch(judged_job, monkeypatch):
    # A corner camera's face is too small in the 360p window; the 1080p stretch make has just
    # fetched holds it. The window still gives the shot cuts.
    from streamclip import framing
    real, asked = framing.plan_layout, []

    def plan(window, window_start, pick, camera, **more):
        asked.append((window, more.get("clip_video")))
        return real(window, window_start, pick, camera, **more)

    monkeypatch.setattr(framing, "plan_layout", plan)
    stages.make(judged_job, only=1)
    ((window, clip),) = asked
    assert window.name == "window.mp4" and clip == judged_job.clip(1) / "source.mp4"
