"""
src/utils/language.py

Replaces BOTH the old language.py and i18n.py. Delete src/utils/i18n.py
after switching imports.

Two responsibilities:
  1. detect_language()  - deterministic, constrained to {pt, en}
  2. to_portuguese()    - translate the RETRIEVAL query into PT

Why (2) exists: the corpus is 100% Portuguese. Measured on the OBG index,
embedding an English query costs you most of the signal:

    query                          top cosine
    como recupero minha senha?        0.592
    how do I recover my password?     0.027   <-- unusable

Answer generation still happens in the user's language. Only the string
that gets embedded is translated.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from langdetect import DetectorFactory, LangDetectException, detect

logger = logging.getLogger("bgo_chatbot.language")

# CRITICAL: langdetect seeds its RNG randomly per process without this.
# Same query -> different language -> different answer across machines.
DetectorFactory.seed = 0

SUPPORTED = ("pt", "en")
DEFAULT_LANG = "pt"  # Brazilian audience; PT is the safe fallback

# Portuguese-Spanish confusion is the main failure mode on short queries,
# so we bias with high-signal PT tokens before trusting langdetect.
_PT_STRONG = re.compile(
    r"\b(voc[eê]|n[aã]o|s[aã]o|est[aã]o|ser[aá]|quais|"
    r"inscri\w*|senha|equipe|prova|escola|professor\w*|aluno\w*|"
    r"estudante\w*|regulamento|orientador\w*|olimp[ií]ada)\b",
    re.IGNORECASE,
)
_EN_STRONG = re.compile(
    r"\b(the|is|are|how|what|when|who|can|does|do|my|password|team|"
    r"school|student|teacher|registration|deadline)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """
    Returns exactly 'pt' or 'en'. Never a surprise code like 'es' or 'ca'.

    Order: strong lexical signal -> langdetect -> DEFAULT_LANG.
    """
    if not text or not text.strip():
        return DEFAULT_LANG

    pt_hits = len(_PT_STRONG.findall(text))
    en_hits = len(_EN_STRONG.findall(text))
    if pt_hits > en_hits:
        return "pt"
    if en_hits > pt_hits and pt_hits == 0:
        return "en"

    try:
        code = detect(text)
    except LangDetectException:
        logger.debug("langdetect failed on %r; defaulting to %s", text, DEFAULT_LANG)
        return DEFAULT_LANG

    if code == "en":
        return "en"
    # es/ca/gl/it/pt and anything else Romance -> treat as PT.
    # The corpus is PT and the users are Brazilian; this is the safe collapse.
    return "pt"


def is_portuguese(text: str) -> bool:
    return detect_language(text) == "pt"


# ---------------------------------------------------------------- translation

_TRANSLATE_PROMPT = (
    "Traduza a pergunta a seguir para portugues do Brasil, usando o "
    "vocabulario oficial de um regulamento de olimpiada escolar "
    "(senha, inscricao, equipe, professor orientador, prova, fase, "
    "certificado, escola, estudante). Mantenha siglas como OBG e iGeo. "
    "Responda APENAS com a traducao, sem aspas, sem explicacao.\n\n"
    "Pergunta: {q}"
)

_translator = None


def _get_translator():
    """Lazy so importing this module doesn't require an API key."""
    global _translator
    if _translator is None:
        from langchain_openai import ChatOpenAI

        _translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _translator


@lru_cache(maxsize=1024)
def to_portuguese(query: str) -> str:
    """
    Returns the query in Portuguese, for embedding only.
    Falls back to the original string on any failure - a slightly worse
    retrieval is much better than a 500.
    """
    if not query or not query.strip():
        return query
    if detect_language(query) == "pt":
        return query

    try:
        resp = _get_translator().invoke(_TRANSLATE_PROMPT.format(q=query))
        translated = (resp.content or "").strip().strip('"')
        if not translated:
            return query
        logger.debug("Query translated for retrieval: %r -> %r", query, translated)
        return translated
    except Exception as e:
        logger.warning("Query translation failed (%s); embedding original", e)
        return query


async def ato_portuguese(query: str) -> str:
    """Async variant for the pipeline's await path."""
    if not query or not query.strip():
        return query
    if detect_language(query) == "pt":
        return query
    try:
        resp = await _get_translator().ainvoke(_TRANSLATE_PROMPT.format(q=query))
        translated = (resp.content or "").strip().strip('"')
        return translated or query
    except Exception as e:
        logger.warning("Query translation failed (%s); embedding original", e)
        return query