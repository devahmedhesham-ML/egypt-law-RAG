"""Corpus records → one bilingual chunk per article, so the top 10 hits are always 10 different articles."""

from __future__ import annotations

from dataclasses import dataclass

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


def chunk_text(r: dict) -> str:
    n = r["article_number"]
    if r["is_repealed"]:  # indexed, so "what does Article 60 say?" retrieves "repealed" instead of a guess
        body = "\n".join(x for x in (r.get("repeal_note_ar"), r.get("repeal_note")) if x)
        lines = [heading_path(r), f"Article {n} | مادة {n} (repealed | ملغاة)", body]
    else:
        lines = [heading_path(r), f"Article {n} | مادة {n}", r["text_ar"], r["text_en"]]
    return "\n".join(x for x in lines if x)


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


def build_chunks(records: list[dict], normalize_arabic: bool = False) -> list[Chunk]:
    chunks = []
    for r in records:
        text = chunk_text(r)
        chunks.append(Chunk(
            id=f"art-{r['article_number']:04d}",
            article_number=r["article_number"],
            text=text,
            embed_text=normalize_ar(text) if normalize_arabic else text,
            metadata=chunk_metadata(r),
        ))
    return chunks


def query_text(question: str, normalize_arabic: bool = False) -> str:
    """Queries get the same Arabic normalization as the chunks they are compared with."""
    return normalize_ar(question) if normalize_arabic else question
