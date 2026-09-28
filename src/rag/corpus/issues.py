"""Warnings and notes raised while building or checking the corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Issue:
    level: str  # "warning": look before you index · "info": expected quirk, recorded for the report
    code: str
    message: str
    article: int | None = None
    page: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def warning(code: str, message: str, article: int | None = None, page: int | None = None) -> Issue:
    return Issue("warning", code, message, article, page)


def info(code: str, message: str, article: int | None = None, page: int | None = None) -> Issue:
    return Issue("info", code, message, article, page)
