"""Corpus records → chunks for the index.

Strategies (params.yaml ingest.chunking):
- article (default): one bilingual chunk per article, so the top 10 hits are always 10 different articles
- window: articles longer than `chunk_size` tokens are split into windows with `overlap` tokens shared
  between neighbours; every window repeats the heading path and article number
- per_language: one Arabic and one English chunk per article, so a query meets text in its own language
Retrieval maps chunks back to articles (best-scoring chunk wins), so answers always cite whole articles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.corpus.hierarchy import AR_ORDINAL_WORD
from rag.corpus.normalize import normalize_ar

# (level, English label, Arabic label) in heading order; topics carry their own Arabic numbering
LEVELS = (
    ("part", "Part", "القسم"),
    ("book", "Book", "الكتاب"),
    ("chapter", "Chapter", "الباب"),
    ("section", "Section", "الفصل"),
    ("topic", "Topic", None),
)


@dataclass(frozen=True)
class Chunk:
    id: str
    article_number: int
    text: str  # stored in Chroma; what people and the LLM read
    embed_text: str  # what gets embedded (Arabic normalized when configured)
    metadata: dict  # Chroma-safe: str/int/float/bool only, no None


def heading_path(r: dict) -> str:
    """'Section 2 Persons | الفصل الثاني الأشخاص > Topic 1 Individuals | الشخص الطبيعي'."""
    parts = []
    for key, label_en, label_ar in LEVELS:
        n, title_en, title_ar = r.get(f"{key}_number"), r.get(f"{key}_title_en"), r.get(f"{key}_title_ar")
        if n is None and not title_en:
            continue
        en = f"{label_en} {n} {title_en or ''}".strip() if n is not None else title_en
        ar = (f"{label_ar} {AR_ORDINAL_WORD.get(n, n)} {title_ar or ''}".strip()
              if label_ar and n is not None else (title_ar or ""))
        parts.append(f"{en} | {ar}" if ar else en)
    if r.get("subtopic_title_en"):
        sub_ar = r.get("subtopic_title_ar")
        parts.append(f"{r['subtopic_title_en']} | {sub_ar}" if sub_ar else r["subtopic_title_en"])
    return " > ".join(parts)


STRATEGIES = ("article", "window", "per_language")


def chunk_header(r: dict) -> str:
    n = r["article_number"]
    label = f"Article {n} | مادة {n}" + (" (repealed | ملغاة)" if r["is_repealed"] else "")
    return "\n".join(x for x in (heading_path(r), label) if x)


def chunk_body(r: dict, lang: str | None = None) -> str:
    """Article text: both languages, or one (`ar` / `en`). Repealed articles carry their repeal note."""
    if r["is_repealed"]:  # indexed, so "what does Article 60 say?" retrieves "repealed" instead of a guess
        ar, en = r.get("repeal_note_ar"), r.get("repeal_note")
    else:
        ar, en = r["text_ar"], r["text_en"]
    parts = {"ar": [ar], "en": [en], None: [ar, en]}[lang]
    return "\n".join(x for x in parts if x)


def chunk_text(r: dict) -> str:
    return "\n".join(x for x in (chunk_header(r), chunk_body(r)) if x)


def chunk_metadata(r: dict) -> dict:
    meta = {
        "article_number": r["article_number"],
        "is_repealed": r["is_repealed"],
        "citation": r["citation"],
        "source_pages": ",".join(map(str, r["source_pages"])),
    }
    for key, *_ in LEVELS:
        for field in ("number", "title_en", "title_ar"):
            value = r.get(f"{key}_{field}")
            if value is not None:
                meta[f"{key}_{field}"] = value
    for field in ("title_en", "title_ar"):
        if r.get(f"subtopic_{field}"):
            meta[f"subtopic_{field}"] = r[f"subtopic_{field}"]
    return meta


def windows(text: str, size: int, overlap: int, tokenizer: Any) -> list[str]:
    """Split text into pieces of at most `size` tokens, `overlap` tokens shared; cut at token boundaries."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) <= size:
        return [text]
    step = size - overlap
    out = []
    for start in range(0, len(offsets), step):
        end = min(start + size, len(offsets))
        out.append(text[offsets[start][0]: offsets[end - 1][1]].strip())
        if end == len(offsets):
            break
    return out


def _chunk(r: dict, suffix: str, header: str, body: str, normalize_arabic: bool, extra: dict) -> Chunk:
    text = "\n".join(x for x in (header, body) if x)
    return Chunk(
        id=f"art-{r['article_number']:04d}{suffix}",
        article_number=r["article_number"],
        text=text,
        embed_text=normalize_ar(text) if normalize_arabic else text,
        metadata={**chunk_metadata(r), **extra},
    )


def build_chunks(records: list[dict], normalize_arabic: bool = False, *, strategy: str = "article",
                 chunk_size: int | None = None, overlap: int = 0, tokenizer: Any = None) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r} (expected one of {STRATEGIES})")
    if strategy == "window":
        if not chunk_size or tokenizer is None:
            raise ValueError("window chunking needs chunk_size and the embedding model's tokenizer")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    chunks = []
    for r in records:
        header = chunk_header(r)
        if strategy == "article":
            chunks.append(_chunk(r, "", header, chunk_body(r), normalize_arabic, {}))
        elif strategy == "per_language" and not r["is_repealed"]:
            for lang in ("ar", "en"):
                chunks.append(_chunk(r, f"-{lang}", header, chunk_body(r, lang), normalize_arabic, {"lang": lang}))
        elif strategy == "per_language":  # a repeal note is one short bilingual chunk
            chunks.append(_chunk(r, "", header, chunk_body(r), normalize_arabic, {}))
        else:
            header_tokens = len(tokenizer(header, add_special_tokens=False)["input_ids"]) + 1
            size = max(chunk_size - header_tokens, overlap + 16)
            pieces = windows(chunk_body(r), size, overlap, tokenizer)
            for i, piece in enumerate(pieces):
                suffix = f"-w{i:02d}" if len(pieces) > 1 else ""
                chunks.append(_chunk(r, suffix, header, piece, normalize_arabic, {"window": i}))
    return chunks


def chunks_per_article(strategy: str) -> int:
    """How many chunks to fetch per wanted article, so k distinct articles survive de-duplication."""
    return 1 if strategy == "article" else 4


def query_text(question: str, normalize_arabic: bool = False) -> str:
    """Queries get the same Arabic normalization as the chunks they are compared with."""
    return normalize_ar(question) if normalize_arabic else question
