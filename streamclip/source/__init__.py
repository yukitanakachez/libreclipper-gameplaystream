"""Where a stream comes from: a Twitch past broadcast or a video file on disk."""

from __future__ import annotations

from urllib.parse import urlparse

from streamclip.errors import Refused
from streamclip.protocols import Source

OTHER_LINK = ("Only Twitch links and video files can be clipped. Send the link of one "
              "finished Twitch broadcast, or a video file.")

# Links pasted without https://, which would otherwise read as a relative path.
BARE_TWITCH = ("twitch.tv/", "www.twitch.tv/", "m.twitch.tv/", "clips.twitch.tv/")

__all__ =["open_source", "is_twitch_host"]


def is_twitch_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return host == "twitch.tv" or host.endswith(".twitch.tv")


def open_source(source: str) -> Source:
    """A TwitchSource for a twitch.tv link, a FileSource for a path. Other links are refused.

    Nothing is fetched here: describe() is where a source is first checked."""
    text = str(source).strip()
    if "://" in text or text.lower().startswith(BARE_TWITCH):
        parsed = urlparse(text if "://" in text else f"https://{text}")
        if parsed.scheme in ("http", "https") and is_twitch_host(parsed.hostname or ""):
            from streamclip.source.twitch import TwitchSource
            return TwitchSource(text if "://" in text else f"https://{text}")
        raise Refused(OTHER_LINK)
    from streamclip.source.local import FileSource
    return FileSource(text)
