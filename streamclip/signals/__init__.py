"""Signals: one excitement value per second of a stream, and the shortlist of hot spots."""

from streamclip.signals.curve import build_signals
from streamclip.signals.events import ensure_model
from streamclip.signals.shortlist import pick_hotspots

__all__ = ["build_signals", "pick_hotspots", "ensure_model"]
