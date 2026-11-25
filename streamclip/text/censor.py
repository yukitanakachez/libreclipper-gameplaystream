"""Strong swear words are starred in captions. The audio is left as it is."""

from __future__ import annotations

_STEMS = {
    "fuck": ["", "s", "ed", "er", "ers", "in", "ing", "ings", "face", "faces", "head", "heads",
             "wit", "wits", "up", "ups", "boy", "boys"],
    "motherfuck": ["er", "ers", "in", "ing", "a", "as"],
    "shit": ["", "s", "ty", "tier", "tiest", "ting", "ted", "head", "heads", "hole", "holes",
             "show", "shows", "face", "talk"],
    "bullshit": ["", "s", "ting", "ted"],
    "horseshit": [""],
    "dipshit": ["", "s"],
    "bitch": ["", "es", "ed", "ing", "y", "ass", "asses"],
    "cunt": ["", "s", "y"],
    "dick": ["", "s", "head", "heads", "hole", "holes", "face", "wad", "wads"],
    "pussy": [""],
    "pussies": [""],
    "asshole": ["", "s"],
}
STRONG = frozenset(stem + end for stem, ends in _STEMS.items() for end in ends)


def censor(word: str) -> str:
    """'Fucking!' -> 'F******!'. Only whole strong words; punctuation and capitals are kept."""
    first = next((i for i, c in enumerate(word) if c.isalnum()), None)
    if first is None:
        return word
    last = max(i for i, c in enumerate(word) if c.isalnum())
    core = word[first:last + 1]
    if core.lower().replace("'", "") not in STRONG:
        return word
    starred = core[0] + "".join("*" if c.isalpha() else c for c in core[1:])
    return word[:first] + starred + word[last + 1:]
