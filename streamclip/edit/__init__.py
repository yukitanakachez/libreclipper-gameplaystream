"""Cut and render: what dead air to cut, where the sounds and the zoom go, and the encode."""

from streamclip.edit.deadair import keep_intervals, motion_series
from streamclip.edit.render import render_clip
from streamclip.edit.sfx import place_sounds, zoom_at

__all__ = ["keep_intervals", "motion_series", "place_sounds", "render_clip", "zoom_at"]
