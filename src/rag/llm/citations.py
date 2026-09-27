"""Extract inline [Article N] citations and check them against the retrieved context."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Tolerates Arabic-Indic digits and "Art."/"Articles" slips from smaller models.
_CITATION = re.compile(r"\[\s*Art(?:icle)?s?\.?\s*([0-9٠-٩]+)\s*\]", re.IGNORECASE)
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass(frozen=True)
class CitationCheck:
    cited: list[int]
    valid: list[int]
    invalid: list[int]  # cited but not in the retrieved context: hallucinated

    @property
    def ok(self) -> bool:
        return not self.invalid


def extract_citations(text: str) -> list[int]:
    seen: dict[int, None] = {}
    for m in _CITATION.finditer(text):
        seen[int(m.group(1).translate(_ARABIC_INDIC))] = None
    return list(seen)


def check_citations(text: str, retrieved: Iterable[int]) -> CitationCheck:
    allowed = set(retrieved)
    cited = extract_citations(text)
    return CitationCheck(
        cited=cited,
        valid=[n for n in cited if n in allowed],
        invalid=[n for n in cited if n not in allowed],
    )
