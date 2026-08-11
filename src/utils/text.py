"""
src/utils/text.py  (was empty - this is why the ligatures survived)

Call clean_pdf_text() on every chunk in
src/rag_pipeline/retrieval/loader.py BEFORE splitting and embedding.

The OBG PDFs come out of extraction like this:

    'solicitamos \nque \na senha \nseja \ngerada \nmanualmente'
    'reconﬁguração', 'classiﬁcadas', 'conﬂitos', 'ﬁns'

Two separate problems:
  - a newline every 2-3 words (PDF layout, not sentence structure)
  - Unicode ligatures U+FB01 (ﬁ) and U+FB02 (ﬂ) as SINGLE characters,
    so 'reconﬁguração' tokenizes as unrelated fragments and the
    embedding for that chunk is measurably degraded.

Rebuild the index after adding this.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures NFKC misses or that appear in these specific PDFs.
_EXTRA_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

_SOFT_HYPHEN = "\u00ad"
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# 'reconfigu-\nração' -> 'reconfiguração'
_HYPHEN_BREAK = re.compile(r"(?<=\w)[-\u2010\u2011]\s*\n\s*(?=\w)")

# A newline that is NOT a paragraph break: single \n between text.
_SOFT_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")

_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")

# Page furniture that adds noise to every chunk's embedding.
_PAGE_NOISE = re.compile(
    r"^\s*(p[áa]gina\s+\d+\s*(de\s+\d+)?|\d+\s*/\s*\d+|\d{1,3})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_pdf_text(text: str) -> str:
    """Normalize extracted PDF text for embedding. Idempotent."""
    if not text:
        return ""

    # NFKC folds most ligatures and full-width forms; the map catches the rest.
    text = unicodedata.normalize("NFKC", text)
    for lig, repl in _EXTRA_LIGATURES.items():
        text = text.replace(lig, repl)

    text = text.replace(_SOFT_HYPHEN, "")
    text = _ZERO_WIDTH.sub("", text)

    # Order matters: rejoin hyphenated words before collapsing newlines.
    text = _HYPHEN_BREAK.sub("", text)
    text = _PAGE_NOISE.sub("", text)
    text = _SOFT_NEWLINE.sub(" ", text)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()


def normalize_for_match(text: str) -> str:
    """
    Accent- and case-insensitive form, for BM25 and keyword matching only.
    Never embed this - the embedder wants the accents.
    """
    text = unicodedata.normalize("NFD", clean_pdf_text(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def assert_clean(text: str) -> None:
    """Use in a unit test to stop dirty text reaching the index again."""
    bad = [c for c in _EXTRA_LIGATURES if c in text]
    assert not bad, f"unnormalized ligatures present: {bad!r}"
    assert _HYPHEN_BREAK.search(text) is None, "hyphenated line break present"