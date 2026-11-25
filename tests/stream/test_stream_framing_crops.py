"""Crop arithmetic for the three layouts. Every rectangle is a fraction of the source frame."""

import pytest

from clipper.models import Rect
from streamclip.framing import crops

WIDE = 16 / 9
CAM = Rect(0.02, 0.03, 0.25, 0.25)                 # a camera box in the top left
FACE = Rect(0.11, 0.08, 0.07, 0.12)                # a face inside it


def pixel_aspect(r, src_aspect):                    # width / height of the crop in pixels
    return (r.w * src_aspect) / r.h


def inside(inner, outer=Rect(0, 0, 1, 1)):
    return (inner.x >= outer.x - 1e-6 and inner.y >= outer.y - 1e-6 and
            inner.x + inner.w <= outer.x + outer.w + 1e-6 and inner.y + inner.h <= outer.y + outer.h + 1e-6)


def test_face_panel_has_the_panel_shape_and_stays_in_the_camera():
    crop = crops.face_panel_crop(FACE, CAM, WIDE)
    assert pixel_aspect(crop, WIDE) == pytest.approx(1080 / 692, rel=0.01) and inside(crop, CAM)
    assert crop.x <= FACE.x and crop.x + crop.w >= FACE.x + FACE.w          # the face is in it


def test_face_panel_is_the_largest_that_fits_the_camera():
    crop = crops.face_panel_crop(FACE, CAM, WIDE)
    inset = crops.CAMERA_INSET
    assert crop.h == pytest.approx(CAM.h * (1 - 2 * inset))                 # a 16:9 box is wider than the panel
    tall = crops.face_panel_crop(FACE, Rect(0.02, 0.03, 0.2, 0.6), WIDE)
    assert tall.w == pytest.approx(0.2 * (1 - 2 * inset))
    assert pixel_aspect(tall, WIDE) == pytest.approx(1080 / 692, rel=0.01)


def test_face_panel_stays_clear_of_the_camera_border():
    # The sample's camera box as the finder sees it: the 4 px light border and a few pixels
    # of the game around it are inside the box, so the crop keeps away from every edge.
    cam = Rect(22 / 1280, 23 / 720, 332 / 1280, 266 / 720)
    picture = Rect(28 / 1280, 28 / 720, 320 / 1280, 256 / 720)   # inside the border
    face = Rect(110 / 1280, 60 / 720, 90 / 1280, 150 / 720)
    crop = crops.face_panel_crop(face, cam, WIDE)
    assert inside(crop, picture)
    assert pixel_aspect(crop, WIDE) == pytest.approx(1080 / 692, rel=0.01)
    for edge in (crop.x - cam.x, cam.x + cam.w - crop.x - crop.w):
        assert edge >= crops.CAMERA_INSET * cam.w - 1e-9
    for edge in (crop.y - cam.y, cam.y + cam.h - crop.y - crop.h):
        assert edge >= crops.CAMERA_INSET * cam.h - 1e-9


def test_a_tiny_camera_box_still_gives_a_panel_shaped_crop():
    cam = Rect(0.4, 0.4, 0.001, 0.001)
    crop = crops.face_panel_crop(Rect(0.4, 0.4, 0.001, 0.001), cam, WIDE)
    assert crop.w > 0 and crop.h > 0 and inside(crop, cam)
    assert pixel_aspect(crop, WIDE) == pytest.approx(1080 / 692, rel=0.01)


def test_face_panel_centres_the_face_in_the_part_the_app_does_not_cover():
    # The top 10% of the finished frame (192 px) is under the app's tabs, and that is the top
    # of the 692 px face panel: the face's centre goes in the middle of the other 500 px.
    visible_middle = (0.10 * 1920 + 692) / 2 / 692
    assert crops.FACE_PANEL_DOWN == pytest.approx(visible_middle)
    cam = Rect(0.1, 0.1, 0.3, 0.8)
    face = Rect(0.2, 0.45, 0.07, 0.12)
    crop = crops.face_panel_crop(face, cam, WIDE)
    assert (face.y + face.h / 2 - crop.y) / crop.h == pytest.approx(visible_middle)
    safe = crops.face_panel_crop(face, None, WIDE)   # narrower than the frame, so free to centre
    assert safe.x + safe.w / 2 == pytest.approx(face.x + face.w / 2)
    assert (face.y + face.h / 2 - safe.y) / safe.h == pytest.approx(visible_middle)


def test_a_face_high_in_its_camera_box_is_moved_down_as_far_as_the_box_allows():
    # The sample: a big face near the top of a box with room to spare above the crop.
    cam = Rect(0.0173, 0.0315, 0.2591, 0.3697)
    face = Rect(0.06, 0.09, 0.15, 0.20)
    crop = crops.face_panel_crop(face, cam, WIDE)
    assert crop.y == pytest.approx(cam.y + crops.CAMERA_INSET * cam.h)


def test_the_zoom_keeps_the_face_where_the_panel_has_it():
    cam = Rect(0.1, 0.1, 0.3, 0.8)
    face = Rect(0.2, 0.45, 0.07, 0.12)
    zoom = crops.zoom_crop(crops.face_panel_crop(face, cam, WIDE), face)
    assert (face.y + face.h / 2 - zoom.y) / zoom.h == pytest.approx(crops.FACE_PANEL_DOWN)


def test_face_panel_without_a_camera_is_a_safe_crop_around_the_face():
    crop = crops.face_panel_crop(FACE, None, WIDE)
    assert pixel_aspect(crop, WIDE) == pytest.approx(1080 / 692, rel=0.01) and inside(crop)
    assert crop.w == pytest.approx(FACE.w * 3.2, rel=0.05)


def test_safe_crop_near_a_corner_is_pushed_back_into_the_frame():
    crop = crops.face_panel_crop(Rect(0.93, 0.0, 0.07, 0.12), None, WIDE)
    assert inside(crop) and crop.x + crop.w == pytest.approx(1.0) and crop.y == 0.0


def test_safe_crop_of_a_huge_face_shrinks_to_the_frame():
    crop = crops.face_panel_crop(Rect(0.2, 0.1, 0.6, 0.8), None, WIDE)
    assert inside(crop) and pixel_aspect(crop, WIDE) == pytest.approx(1080 / 692, rel=0.01)


def test_game_panel_is_full_height_and_about_half_the_width():
    crop = crops.game_panel_crop(None, WIDE)
    assert (crop.y, crop.h) == (0.0, 1.0) and crop.w == pytest.approx(0.494, abs=0.005)
    assert crop.x == pytest.approx((1 - crop.w) / 2)


def test_game_panel_slides_away_from_a_camera_it_would_show_twice():
    cam = Rect(0.30, 0.0, 0.25, 0.25)                # overlaps the centred crop
    crop = crops.game_panel_crop(cam, WIDE)
    assert crop.x > (1 - crop.w) / 2 and inside(crop)


def test_game_panel_slides_left_when_that_hides_more_of_the_camera():
    crop = crops.game_panel_crop(Rect(0.50, 0.7, 0.25, 0.25), WIDE)
    assert crop.x < (1 - crop.w) / 2 and inside(crop)
    assert crop.x + crop.w == pytest.approx(0.50)    # just clear of the camera, no further


def test_game_panel_ignores_a_camera_it_does_not_show():
    corner = Rect(0.02, 0.03, 0.20, 0.25)            # ends before the centred crop begins
    assert crops.game_panel_crop(corner, WIDE).x == pytest.approx((1 - 0.494) / 2, abs=0.005)


def test_game_panel_slides_clear_of_a_sliver_of_the_camera():
    # The usual stream: a camera a quarter of the frame wide in the top-left corner. Centred,
    # the game panel showed its last 30 pixels, border and all, down its left edge.
    cam = Rect(0.017, 0.032, 0.259, 0.370)
    crop = crops.game_panel_crop(cam, WIDE)
    assert crop.x == pytest.approx(cam.x + cam.w) and inside(crop)


def test_game_panel_keeps_the_middle_when_clearing_a_little_camera_would_cost_a_lot():
    cam = Rect(0.0, 0.0, 0.33, 0.30)                 # under a quarter of it shows; clearing it is a long slide
    assert crops.game_panel_crop(cam, WIDE).x == pytest.approx((1 - 0.494) / 2, abs=0.005)


def test_narrow_sources_keep_their_width():
    crop = crops.game_panel_crop(None, 4 / 5)        # a source narrower than the panel
    assert (crop.x, crop.w) == (0.0, 1.0) and pixel_aspect(crop, 4 / 5) == pytest.approx(1080 / 1228, rel=0.01)
    assert inside(crop)


def test_full_frame_crops_are_nine_by_sixteen():
    for crop in (crops.full_game_crop(WIDE), crops.full_face_crop(Rect(0.8, 0.2, 0.15, 0.3), WIDE)):
        assert pixel_aspect(crop, WIDE) == pytest.approx(9 / 16, rel=0.01) and inside(crop)


def test_full_game_crop_is_centred_and_full_face_crop_follows_the_face():
    game = crops.full_game_crop(WIDE)
    assert game.x == pytest.approx((1 - game.w) / 2) and (game.y, game.h) == (0.0, 1.0)
    face = Rect(0.8, 0.2, 0.15, 0.3)
    assert crops.full_face_crop(face, WIDE).x > game.x


def test_zoom_is_tighter_same_shape_and_inside_the_crop():
    base = crops.face_panel_crop(FACE, CAM, WIDE)
    zoom = crops.zoom_crop(base, FACE)
    assert zoom.w == pytest.approx(base.w / 1.35, rel=0.01) and inside(zoom, base)
    assert pixel_aspect(zoom, WIDE) == pytest.approx(pixel_aspect(base, WIDE), rel=0.01)


def test_zoom_on_a_face_at_the_edge_stays_inside_the_crop():
    base = crops.full_face_crop(Rect(0.0, 0.0, 0.2, 0.4), WIDE)
    assert inside(crops.zoom_crop(base, Rect(0.0, 0.0, 0.2, 0.4)), base)
