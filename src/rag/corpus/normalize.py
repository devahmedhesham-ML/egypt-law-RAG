"""Arabic normalization for search. Apply the same function to documents and to queries."""

from __future__ import annotations

import re

_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alef, tatweel
_ALEF = re.compile(r"[أإآٱ]")


def normalize_ar(text: str) -> str:
    """Strip diacritics and tatweel, unify alef variants and alef maqsura, so spellings embed alike."""
    return _ALEF.sub("ا", _DIACRITICS.sub("", text)).replace("ى", "ي")
