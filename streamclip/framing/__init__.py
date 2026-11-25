"""Framing: where the camera sits in the frame, and what each shot of a clip crops.

`find_camera` runs once per stream, `plan_layout` once per clip. Every rectangle is a fraction
of the source frame, so a layout worked out on a 360p window applies to the 1080p stretch.
"""

from streamclip.framing.camera import find_camera
from streamclip.framing.layout import plan_layout

__all__ = ["find_camera", "plan_layout"]
