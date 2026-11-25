"""Stage 1, steps 6 and 7: each hot spot's window transcribed, fetched small and laid out on
one contact sheet, under `candidates/NN/`."""

from streamclip.windows.build import build_candidates
from streamclip.windows.sheet import contact_sheet
from streamclip.windows.transcribe import (
    captions_allowed, ensure_model, transcribe_clip, transcribe_window,
)

__all__ = ["build_candidates", "captions_allowed", "contact_sheet", "ensure_model",
           "transcribe_clip", "transcribe_window"]
