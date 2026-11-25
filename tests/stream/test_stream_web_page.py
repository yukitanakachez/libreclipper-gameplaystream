"""The test page itself: one self-contained file served at /."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient


def create_app(jobs_root):
    """Imported when called, never at collection: Lane A's serve test checks the page is loaded
    only when asked, so no test module may import it up front."""
    from streamclip.web.app import create_app as make
    return make(jobs_root)


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path))


def test_the_page_is_served_and_self_contained(client):
    response = client.get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    html = response.text
    for part in ('id="source"', 'id="file"', 'id="clips"', 'id="run"', 'id="stages"', 'id="curve"',
                 'id="candidates"', 'id="clips-out"'):
        assert part in html
    assert "http://" not in html and "https://" not in html.replace("https://www.twitch.tv/videos/", "")


def test_no_text_is_smaller_than_twelve_pixels(client):
    sizes = re.findall(r"font-size:\s*([\d.]+)(px|rem)", client.get("/").text)
    assert sizes and all(float(n) >= (12 if unit == "px" else 0.75) for n, unit in sizes)


def test_the_page_loads_nothing_from_elsewhere(client):
    html = client.get("/").text
    assert not re.search(r"<(script|link)\b[^>]*\b(src|href)=", html)             # styles and code inline
    assert not re.search(r"\b(src|href)=\"(https?:)?//", html)                       # nothing off this server
    assert "@import" not in html and "url(" not in html


def test_the_clip_count_box_allows_one_to_ten(client):
    html = client.get("/").text
    box = re.search(r"<input[^>]*id=\"clips\"[^>]*>", html).group(0)
    assert 'min="1"' in box and 'max="10"' in box and 'value="6"' in box
