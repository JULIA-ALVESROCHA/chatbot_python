"""
src/rag_pipeline/retrieval/text_splitter.py

Item-aware splitting for the OBG corpus.

WHY: RecursiveCharacterTextSplitter(500, 50) cuts wherever the counter
lands. Real chunks from the current index:

    "treinamento dos estudantes selecionados.    2. DA PARTICIPAÇ..."
    "regulamento.    3. Limite de participantes    3.1  Embora  p..."
    "e a que invalida o pedido anterior. Assim, recomendamos que..."

Each starts mid-sentence with an orphan fragment of the previous rule, and
the first is truncated mid-word. The embedding of such a chunk is a blend of
two unrelated rules, which is why top cosine tops out at 0.744 instead of
0.85+.

The regulamento is a NUMBERED document: 1.7, 2.4.2, 4.4.2. Each item is one
self-contained rule and matches how the gold answers are written. Splitting
on those boundaries keeps whole rules together.

Measured problem this should fix: "Quem pode ser professor orientador" has
5 gold facts (corpo docente, estagiarios, plantonistas, coordenadores,
vinculo) in ONE item, currently spread over 3-4 chunks -> recall 3.3%.

Support PDFs (Duvidas, Procedimentos) are NOT numbered - they fall back to
heading/paragraph splitting.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.app.core.config import settings
from src.utils.text import clean_pdf_text, clean_pdf_text_structural

logger = logging.getLogger("bgo_chatbot.splitter")

# A numbered item at the start of a line: "1.7", "2.4.2", "4.4.2.", "1.1.3"
_ITEM_START = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){1,3})[.\s)]\s*", re.MULTILINE)

# Top-level section headings: "2. DA PARTICIPACAO", "3. Limite de participantes"
_SECTION = re.compile(r"^\s*(\d{1,2})\.\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{3,80})$", re.MULTILINE)

# Emoji/bold headings in the support PDFs: "🔐 Problemas com Senha Recuperada"
_SUPPORT_HEADING = re.compile(
    r"^\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]\s*)?"
    r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{8,90})$",
    re.MULTILINE,
)

MAX_CHARS = 1200   # a long item still fits; only genuinely huge ones split
MIN_CHARS = 120    # below this, merge forward rather than emit a stub


def _is_numbered_doc(text: str) -> bool:
    """Regulamento and editais are numbered; support PDFs are not."""
    return len(_ITEM_START.findall(text)) >= 5


def _current_section(text: str, pos: int) -> Optional[str]:
    """Nearest section heading above `pos`, for metadata and context."""
    last = None
    for m in _SECTION.finditer(text, 0, pos):
        last = f"{m.group(1)}. {m.group(2).strip()}"
    return last


def _split_numbered(text: str) -> List[dict]:
    """One chunk per numbered item, with section heading prepended."""
    marks = list(_ITEM_START.finditer(text))
    if not marks:
        return []

    pieces: List[dict] = []

    # Anything before the first item (preamble) becomes its own piece.
    if marks[0].start() > MIN_CHARS:
        pieces.append({"item": None, "section": None,
                       "text": text[: marks[0].start()].strip()})

    for i, m in enumerate(marks):
        start = m.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        pieces.append({
            "item": m.group(1),
            "section": _current_section(text, start),
            "text": body,
        })

    # Merge stubs forward: "1.7.1" alone is meaningless without "1.7".
    merged: List[dict] = []
    for p in pieces:
        if merged and len(p["text"]) < MIN_CHARS:
            merged[-1]["text"] += "\n" + p["text"]
        elif merged and len(merged[-1]["text"]) < MIN_CHARS:
            merged[-1]["text"] += "\n" + p["text"]
            merged[-1]["item"] = merged[-1]["item"] or p["item"]
        else:
            merged.append(p)

    # Only oversized items get further split, and on sentence boundaries.
    out: List[dict] = []
    overflow = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        keep_separator=True,
    )
    for p in merged:
        if len(p["text"]) <= MAX_CHARS:
            out.append(p)
            continue
        for j, sub in enumerate(overflow.split_text(p["text"])):
            out.append({"item": p["item"], "section": p["section"],
                        "text": sub, "part": j + 1})
    return out


def _split_prose(text: str) -> List[dict]:
    """Support PDFs: split on headings, then on paragraphs if still too long."""
    heads = list(_SUPPORT_HEADING.finditer(text))
    blocks: List[dict] = []
    if heads:
        # Anything before the first heading (preamble) becomes its own
        # block. Without this, text preceding the first matched heading is
        # silently dropped from the index instead of falling into a block.
        preamble = text[: heads[0].start()].strip()
        if preamble:
            blocks.append({"item": None, "section": None, "text": preamble})
        for i, m in enumerate(heads):
            start = m.start()
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = text[start:end].strip()
            if body:
                blocks.append({"item": None, "section": m.group(1).strip(),
                               "text": body})
    if not blocks:
        blocks = [{"item": None, "section": None, "text": text}]

    # Merge stubs forward: _SUPPORT_HEADING also matches short data lines
    # that aren't real section headings (e.g. "Valor da Inscrição por
    # Equipe:"), producing blocks too short to search well on their own.
    # Folding them into the next block keeps the content instead of the
    # old behavior, which silently dropped anything under MIN_CHARS.
    merged: List[dict] = []
    for b in blocks:
        if merged and len(b["text"]) < MIN_CHARS:
            merged[-1]["text"] += "\n" + b["text"]
        elif merged and len(merged[-1]["text"]) < MIN_CHARS:
            merged[-1]["text"] += "\n" + b["text"]
            merged[-1]["section"] = merged[-1]["section"] or b["section"]
        else:
            merged.append(b)
    blocks = merged

    out: List[dict] = []
    overflow = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "; ", " "],
        keep_separator=True,
    )
    for b in blocks:
        if len(b["text"]) <= MAX_CHARS:
            out.append(b)
        else:
            for j, sub in enumerate(overflow.split_text(b["text"])):
                out.append({**b, "text": sub, "part": j + 1})
    return out


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Drop-in for whatever loader.py currently calls.

    IMPORTANT: boundary detection (_is_numbered_doc / _split_numbered /
    _split_prose) runs on clean_pdf_text_structural() output, which keeps
    single line breaks intact - clean_pdf_text() collapses those to spaces
    and would erase the '^' line starts the boundary regexes need. Each
    resulting piece is then passed through clean_pdf_text() individually
    before being stored, so the final chunk text is still fully normalized.
    """
    chunks: List[Document] = []

    for doc in docs:
        structural = clean_pdf_text_structural(doc.page_content)
        if not structural:
            continue

        pieces = _split_numbered(structural) if _is_numbered_doc(structural) else _split_prose(structural)
        if not pieces:
            pieces = [{"item": None, "section": None, "text": structural}]

        source = (doc.metadata or {}).get("source", "?")
        page = (doc.metadata or {}).get("page", 0)

        for idx, p in enumerate(pieces):
            body = clean_pdf_text(p["text"]).strip()
            if not body:
                continue

            # Prefix the section heading so the embedding carries topic
            # context. "1.7.1 Somente o professor orientador podera..." is
            # ambiguous alone; with "Da Substituicao de Membros da Equipe"
            # above it, it is not.
            embed_text = body
            if p.get("section") and not body.startswith(p["section"]):
                embed_text = f"{p['section']}\n{body}"

            meta = dict(doc.metadata or {})
            meta.update({
                "chunk_id": f"{_slug(source)}_p{page}_i{idx}",
                "item": p.get("item"),
                "section": p.get("section"),
                "part": p.get("part"),
                "n_chars": len(embed_text),
            })
            chunks.append(Document(page_content=embed_text, metadata=meta))

    _log_stats(chunks)
    return chunks


def _slug(path: str) -> str:
    name = re.sub(r"\.pdf$", "", str(path).split("/")[-1], flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _log_stats(chunks: List[Document]) -> None:
    if not chunks:
        logger.warning("Splitter produced 0 chunks")
        return
    sizes = [len(c.page_content) for c in chunks]
    with_item = sum(1 for c in chunks if c.metadata.get("item"))
    logger.info(
        "Split into %d chunks | chars min=%d median=%d max=%d | %d carry an item number",
        len(chunks), min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes), with_item,
    )
    # Cheap regression guard: a chunk that starts lowercase is usually a
    # mid-sentence cut, which is exactly what we are trying to eliminate.
    orphans = [c for c in chunks if c.page_content[:1].islower()]
    if len(orphans) > len(chunks) * 0.15:
        logger.warning(
            "%d/%d chunks start mid-sentence - check the split boundaries",
            len(orphans), len(chunks),
        )