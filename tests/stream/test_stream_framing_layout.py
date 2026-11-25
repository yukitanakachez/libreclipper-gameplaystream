"""The layout of one clip: split, face or game for every shot, in source-clip time."""

import pytest

from clipper.models import Rect
from streamclip.framing import crops, find_camera, plan_layout
from streamclip.schema import CameraBox, Layout, Pick

CAMERA = CameraBox(Rect(0.02, 0.03, 0.25, 0.25), 0.9)
SMALL_IN_BOX = Rect(0.11, 0.08, 0.07, 0.12)
SMALL_ELSEWHERE = Rect(0.70, 0.60, 0.07, 0.12)
BIG = Rect(0.30, 0.10, 0.35, 0.60)
WIDE = 16 / 9


@pytest.fixture
def window(synthetic_stream):
    return synthetic_stream                           # 1280x720; stands in for a candidate's window


def pick(start, end, rank=1):
    return Pick(rank=rank, candidate=3, start=start, end=end, score=8, reason="", title="", caption="",
                hashtags=[], beats=[], protect=[], weak=False)


def faces_between(box, since, until):
    """A face detector that sees `box` on every sample from `since` up to `until` (window seconds)."""
    def detect(video, start, end, fps=10.0):
        times = [start + i / fps for i in range(round((end - start) * fps))]
        return [(t, box, 0.0) for t in times if since <= t < until]
    return detect


def faces_at(box):
    return faces_between(box, 0.0, 1e9)


def faces_until(box, window_second):
    return faces_between(box, 0.0, window_second)


def no_faces(video, start, end, fps=10.0):
    return []


def spy(seen):
    def detect(video, start, end, fps=10.0):
        seen.update(video=video, start=start, end=end, fps=fps)
        return []
    return detect


def one_shot(video, start, end):
    return [(start, end)]


def cut_at(*cuts):
    def shots(video, start, end):
        bounds = [start, *cuts, end]
        return list(zip(bounds, bounds[1:]))
    return shots


def test_small_face_in_the_camera_box_is_a_split(window):
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=faces_at(SMALL_IN_BOX), shots=one_shot)
    (shot,) = layout.shots
    assert (shot.start, shot.end, shot.kind) == (0.0, 40.0, "split")
    assert shot.face and shot.game and shot.zoom and layout.clip == 1


def test_a_split_uses_the_crops_for_the_camera_box(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=faces_at(SMALL_IN_BOX), shots=one_shot).shots
    face = crops.face_panel_crop(SMALL_IN_BOX, CAMERA.rect, WIDE)
    for got, want in ((shot.face, face), (shot.game, crops.game_panel_crop(CAMERA.rect, WIDE)),
                      (shot.zoom, crops.zoom_crop(face, SMALL_IN_BOX))):
        assert [getattr(got, k) for k in "xywh"] == pytest.approx([getattr(want, k) for k in "xywh"], abs=1e-4)


def test_times_are_in_source_clip_time(window):
    seen = {}
    plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=spy(seen), shots=one_shot)
    assert (seen["start"], seen["end"]) == (20.0, 60.0)          # the window's own clock
    assert seen["video"] == window and seen["fps"] == 5.0


def test_the_clip_number_is_the_picks_rank(window):
    assert plan_layout(window, 540.0, pick(560.0, 600.0, rank=4), CAMERA, detect=no_faces, shots=one_shot).clip == 4


def test_a_big_face_fills_the_frame(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=faces_at(BIG), shots=one_shot).shots
    assert shot.kind == "face" and shot.game is None and shot.zoom


def test_no_face_means_the_game_fills_the_frame(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=no_faces, shots=one_shot).shots
    assert shot.kind == "game" and shot.face is None and shot.zoom is None
    assert shot.game.x == pytest.approx(crops.full_game_crop(WIDE).x, abs=1e-4)


def test_a_face_seen_in_under_half_of_a_shot_does_not_count(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA,
                          detect=faces_until(SMALL_IN_BOX, 35.0), shots=one_shot).shots
    assert shot.kind == "game"


def test_no_camera_box_still_splits_around_the_face(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), None, detect=faces_at(SMALL_IN_BOX), shots=one_shot).shots
    assert shot.kind == "split" and shot.face.w == pytest.approx(SMALL_IN_BOX.w * 3.2, rel=0.05)


def test_a_face_outside_the_camera_box_gets_the_safe_crop(window):        # the camera finder was wrong
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=faces_at(SMALL_ELSEWHERE), shots=one_shot).shots
    assert shot.kind == "split" and shot.face.x > CAMERA.rect.x + CAMERA.rect.w


def test_layout_changes_only_at_shot_cuts(window):
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA,
                         detect=faces_until(SMALL_IN_BOX, window_second=40.0), shots=cut_at(40.0))
    assert [(s.start, s.end, s.kind) for s in layout.shots] == [(0.0, 20.0, "split"), (20.0, 40.0, "game")]


def test_neighbouring_shots_with_the_same_layout_are_joined(window):
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=faces_at(SMALL_IN_BOX), shots=cut_at(40.0))
    assert len(layout.shots) == 1


def test_neighbouring_shots_of_one_kind_with_different_crops_stay_apart(window):
    def detect(video, start, end, fps=10.0):
        return (faces_between(SMALL_IN_BOX, 0.0, 40.0)(video, start, end, fps)
                + faces_between(SMALL_ELSEWHERE, 40.0, 1e9)(video, start, end, fps))
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=detect, shots=cut_at(40.0))
    assert [s.kind for s in layout.shots] == ["split", "split"]
    assert layout.shots[0].face != layout.shots[1].face


def test_a_flicker_of_a_shot_takes_its_neighbours_layout(window):
    def detect(video, start, end, fps=10.0):          # no face during the 0.3 s flash
        return [d for d in faces_at(SMALL_IN_BOX)(video, start, end, fps) if not 40.0 <= d[0] < 40.3]
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=detect, shots=cut_at(40.0, 40.3))
    assert [(s.start, s.end, s.kind) for s in layout.shots] == [(0.0, 40.0, "split")]


def test_a_flicker_between_two_layouts_joins_the_longer_side(window):
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA,
                         detect=faces_between(SMALL_IN_BOX, 30.3, 1e9), shots=cut_at(30.0, 30.3))
    assert [(s.start, s.end, s.kind) for s in layout.shots] == [(0.0, 10.0, "game"), (10.0, 40.0, "split")]


def test_shots_cover_the_whole_clip_with_no_gaps(window):
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=faces_until(SMALL_IN_BOX, 33.3), shots=cut_at(33.3, 47.1))
    assert layout.shots[0].start == 0.0 and layout.shots[-1].end == 40.0
    assert all(a.end == b.start for a, b in zip(layout.shots, layout.shots[1:]))


def test_shots_that_spill_past_the_pick_or_leave_gaps_are_mended(window):
    ragged = lambda video, start, end: [(start - 5.0, 30.0), (31.0, end + 9.0)]
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=no_faces, shots=ragged)
    assert [(s.start, s.end) for s in layout.shots] == [(0.0, 40.0)]


def test_every_crop_stays_inside_the_frame(window):
    for box in (SMALL_IN_BOX, SMALL_ELSEWHERE, BIG, Rect(0.93, 0.0, 0.07, 0.12)):
        (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), None, detect=faces_at(box), shots=one_shot).shots
        for r in (shot.face, shot.game, shot.zoom):
            assert r is None or (r.x >= 0 and r.y >= 0 and r.x + r.w <= 1 + 1e-6 and r.y + r.h <= 1 + 1e-6)


def test_a_layout_survives_a_trip_through_json(window):
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA,
                         detect=faces_until(SMALL_IN_BOX, 40.0), shots=cut_at(40.0))
    assert Layout.from_dict(layout.to_dict()) == layout


def real_look(monkeypatch, detect):
    """Puts `detect` where the real face finder is, and one shot where the shot finder is."""
    from clipper.reframe import shots
    from streamclip.framing import smallfaces
    monkeypatch.setattr(smallfaces, "detect_faces", detect)
    monkeypatch.setattr(shots, "detect_shots", one_shot)


def inside(box, region):
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    return region.x <= cx <= region.x + region.w and region.y <= cy <= region.y + region.h


def test_the_real_detectors_are_looked_up_when_called(monkeypatch, window):
    real_look(monkeypatch, lambda video, start, end, fps=10.0, **more: faces_at(BIG)(video, start, end, fps))
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA).shots
    assert shot.kind == "face"


def test_the_real_look_is_inside_the_camera_as_well_as_at_the_whole_frame(monkeypatch, window):
    looks = []

    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        looks.append((fps, tiles, list(regions)))
        return faces_at(SMALL_IN_BOX)(video, start, end, fps)

    real_look(monkeypatch, detect)
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA).shots
    assert shot.kind == "split" and looks == [(5.0, 0, [CAMERA.rect])]


def moved_camera(looks):
    """A face finder for a clip whose camera is at SMALL_ELSEWHERE: seen only in parts of the
    frame, or in a region that holds it. The stream's own camera box shows nothing here."""
    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        looks.append("search" if tiles else "regions" if regions else "whole")
        if tiles or any(inside(SMALL_ELSEWHERE, region) for region in regions):
            return faces_at(SMALL_ELSEWHERE)(video, start, end, fps)
        return []
    return detect


def test_a_clip_whose_camera_box_shows_no_face_is_searched_on_its_own(monkeypatch, window):
    looks = []
    real_look(monkeypatch, moved_camera(looks))
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA).shots
    assert shot.kind == "split" and inside(SMALL_ELSEWHERE, shot.face)
    assert looks == ["regions", "search", "regions"]


def test_a_stream_with_no_camera_found_still_gets_each_clip_s_own(monkeypatch, window):
    looks = []
    real_look(monkeypatch, moved_camera(looks))
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), None).shots
    assert shot.kind == "split" and inside(SMALL_ELSEWHERE, shot.face)
    assert looks == ["whole", "search", "regions"]


def test_no_face_anywhere_is_searched_once_and_stays_a_game_clip(monkeypatch, window):
    looks = []
    real_look(monkeypatch, lambda video, start, end, fps=10.0, tiles=0, regions=(): looks.append(tiles) or [])
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), None).shots
    assert shot.kind == "game" and len(looks) == 2 and looks[1] > 0


def test_a_face_seen_through_the_whole_clip_needs_no_search(monkeypatch, window):
    looks = []

    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        looks.append(tiles)
        return faces_at(SMALL_ELSEWHERE)(video, start, end, fps)

    real_look(monkeypatch, detect)
    plan_layout(window, 540.0, pick(560.0, 600.0), None)
    assert looks == [0]


def test_the_package_hands_out_both_entry_points():
    assert callable(find_camera) and callable(plan_layout)


def test_a_face_hidden_for_a_moment_does_not_end_the_split(window):
    # A hand over the face for two seconds, between two cuts: the camera is still there. Seen
    # on a real clip, which dropped to the game alone for 1.9 s and came back.
    def detect(video, start, end, fps=10.0):
        times = [start + i / fps for i in range(round((end - start) * fps))]
        return [(t, SMALL_IN_BOX, 0.0) for t in times if not 34.0 <= t < 36.0]

    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=detect, shots=cut_at(34.0, 36.0)).shots
    assert shot.kind == "split" and (shot.start, shot.end) == (0.0, 40.0)


def test_a_face_gone_for_longer_still_becomes_a_game_shot(window):
    def detect(video, start, end, fps=10.0):
        times = [start + i / fps for i in range(round((end - start) * fps))]
        return [(t, SMALL_IN_BOX, 0.0) for t in times if not 30.0 <= t < 36.0]

    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=detect, shots=cut_at(30.0, 36.0))
    assert [s.kind for s in layout.shots] == ["split", "game", "split"]


LEFT_IN_BOX = Rect(0.04, 0.08, 0.07, 0.12)
RIGHT_IN_BOX = Rect(0.17, 0.10, 0.07, 0.12)


def here_and_there(first, second):
    """A face at `first` for 45% of the looks, at `second` for 45%, and unseen for the rest."""
    def detect(video, start, end, fps=10.0):
        out = []
        for i in range(round((end - start) * fps)):
            if i % 20 < 18:
                out.append((start + i / fps, first if i % 20 < 9 else second, 0.0))
        return out
    return detect


def test_a_streamer_who_moves_about_inside_the_camera_is_still_on_camera(window):
    # Leaning from one side of the camera to the other makes two faces of one person, each seen
    # in under half the shot. Seen on a real clip, which lost its face panel.
    detect = here_and_there(LEFT_IN_BOX, RIGHT_IN_BOX)
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=detect, shots=one_shot).shots
    assert shot.kind == "split" and shot.face is not None


def test_faces_in_two_places_with_no_camera_are_not_one_person(window):
    detect = here_and_there(LEFT_IN_BOX, RIGHT_IN_BOX)
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), None, detect=detect, shots=one_shot).shots
    assert shot.kind == "game"


def test_a_face_outside_the_camera_does_not_count_as_the_streamer_on_camera(window):
    # In the box for 35% of the shot (under ON_CAMERA), elsewhere for 45% (under half): the
    # two do not add up to the streamer on camera.
    def detect(video, start, end, fps=10.0):
        n = round((end - start) * fps)
        return [(start + i / fps, LEFT_IN_BOX if i % 20 < 7 else SMALL_ELSEWHERE, 0.0)
                for i in range(n) if i % 20 < 16]

    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=detect, shots=one_shot).shots
    assert shot.kind == "game"


# --- Faces read from the clip's own stretch ---------------------------------------------------
# A corner camera's face is 30 to 36 px tall in the 360p window, and the face model misses it
# in most samples (Starwish run 10, clips 2, 4 and 5). make has already fetched the pick's
# stretch at 1080p by the time it frames the clip, so the faces are read from that.

def clip_of(tmp_path):
    clip = tmp_path / "source.mp4"              # t=0 at pick.start; never opened by these fakes
    clip.write_bytes(b"")
    return clip


def test_faces_are_read_from_the_clip_s_own_stretch_when_it_is_given(monkeypatch, window, tmp_path):
    clip, looks = clip_of(tmp_path), []

    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        looks.append((video, start, end))
        return faces_at(SMALL_IN_BOX)(video, start, end, fps)

    real_look(monkeypatch, detect)
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, clip_video=clip).shots
    assert shot.kind == "split" and looks == [(clip, 0.0, 40.0)]


def test_the_clip_s_faces_line_up_with_the_window_s_shot_cuts(monkeypatch, window, tmp_path):
    # The face is in the clip's first 20 s (stream 560-580); the window cuts at its own 40 s,
    # which is stream 580. So the first shot is a split and the second is the game.
    from clipper.reframe import shots
    clip = clip_of(tmp_path)

    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        assert video == clip
        return faces_between(SMALL_IN_BOX, 0.0, 20.0)(video, start, end, fps)

    real_look(monkeypatch, detect)
    monkeypatch.setattr(shots, "detect_shots", cut_at(40.0))
    layout = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, clip_video=clip)
    assert [(s.kind, s.start, s.end) for s in layout.shots] == [("split", 0.0, 20.0), ("game", 20.0, 40.0)]


def test_the_shot_cuts_still_come_from_the_window(monkeypatch, window, tmp_path):
    from clipper.reframe import shots
    clip, asked = clip_of(tmp_path), []
    real_look(monkeypatch, lambda video, start, end, fps=10.0, tiles=0, regions=(): [])
    monkeypatch.setattr(shots, "detect_shots", lambda video, start, end: asked.append((video, start, end)) or [(start, end)])
    plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, clip_video=clip)
    assert asked == [(window, 20.0, 60.0)]


def test_a_missing_clip_stretch_falls_back_to_the_window(monkeypatch, window, tmp_path):
    looks = []

    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        looks.append((video, start, end))
        return faces_at(SMALL_IN_BOX)(video, start, end, fps)

    real_look(monkeypatch, detect)
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, clip_video=tmp_path / "none.mp4").shots
    assert shot.kind == "split" and looks == [(window, 20.0, 60.0)]


def test_a_clip_s_own_search_reads_its_stretch_too(monkeypatch, window, tmp_path):
    # The second camera (on the left while the corner one is off screen) is reached only by
    # the clip's own search, so the search reads the better pixels as well.
    clip, looks, videos = clip_of(tmp_path), [], []
    search = moved_camera(looks)

    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        videos.append(video)
        return search(video, start, end, fps, tiles, regions)

    real_look(monkeypatch, detect)
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, clip_video=clip).shots
    assert shot.kind == "split" and inside(SMALL_ELSEWHERE, shot.face)
    assert looks == ["regions", "search", "regions"] and videos == [clip, clip, clip]


# --- The stream's own camera box needs a face in fewer samples ---------------------------------
# The box was found steady across the stream, so the camera is on screen even while the face
# model misses the face or the streamer looks away. Run 10's clip 4: a face in 34 of the last
# shot's 71 samples, the streamer standing up for the first six seconds of it.

def seen_for(box, share):
    """A face at `box` in the first `share` of every ten samples."""
    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        n = round((end - start) * fps)
        return [(start + i / fps, box, 0.0) for i in range(n) if i % 10 < round(share * 10)]
    return detect


def test_a_face_in_the_stream_s_camera_box_for_two_fifths_of_a_shot_is_a_split(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=seen_for(SMALL_IN_BOX, 0.4),
                          shots=one_shot).shots
    assert shot.kind == "split"


def test_a_face_in_the_stream_s_camera_box_for_a_fifth_of_a_shot_is_the_game(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA, detect=seen_for(SMALL_IN_BOX, 0.2),
                          shots=one_shot).shots
    assert shot.kind == "game"


def test_a_face_outside_any_camera_box_still_needs_half_the_shot(window):
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), None, detect=seen_for(SMALL_ELSEWHERE, 0.4),
                          shots=one_shot).shots
    assert shot.kind == "game"


def test_a_clip_s_own_box_still_needs_half_the_shot(monkeypatch, window):
    # A box made from one clip's search is a guess for that clip, not a camera seen across the
    # stream, so it gets no allowance.
    def detect(video, start, end, fps=10.0, tiles=0, regions=()):
        if tiles:
            return faces_at(SMALL_ELSEWHERE)(video, start, end, fps)
        if any(inside(SMALL_ELSEWHERE, region) for region in regions):
            return seen_for(SMALL_ELSEWHERE, 0.4)(video, start, end, fps)
        return []

    real_look(monkeypatch, detect)
    (shot,) = plan_layout(window, 540.0, pick(560.0, 600.0), CAMERA).shots
    assert shot.kind == "game"
