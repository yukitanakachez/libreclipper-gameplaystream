"""What can go wrong, in words a customer can read."""

from __future__ import annotations


class Refused(Exception):
    """The request cannot be clipped. str(e) is a plain reason for the customer."""


class StageError(Exception):
    """A stage failed. str(e) is a plain reason."""


class NoVideo(StageError):
    """A fetched stretch has no video track."""
