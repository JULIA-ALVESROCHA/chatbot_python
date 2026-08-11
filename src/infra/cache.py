"""
src/infra/cache.py - chat history for conversation sessions

Public API is unchanged: add_to_history, get_history, clear_history,
get_history_list, get_session_count. Drop-in replacement.

What changed and why:

1. THREAD SAFETY. FastAPI serves concurrently; a bare dict mutated from
   several tasks can lose writes. All access is now under a lock.

2. SHARED STORAGE. The old dict was process-local. With uvicorn --workers N
   each worker had its own history, so a follow-up question ("e na fase
   presencial?") could land on a worker with no history, fail to be rewritten
   into a standalone query, retrieve nothing, and answer "nao encontrei".
   Same machine, same index, inconsistent answers.

   Set HISTORY_BACKEND=file (default) to share via disk, or =memory to keep
   the old behaviour when you know you run a single worker.

   For real production use Redis - the file backend is fine for the OBG's
   traffic but it is last-write-wins under concurrency.

3. BOUNDED. Sessions cap at MAX_TURNS_PER_SESSION; total sessions cap at
   MAX_SESSIONS with oldest-first eviction. The old version grew forever.

NOTE: this file does NOT cache answers. Something else in the repo writes
.lumie_cache.json - find it with:
    grep -rn "lumie_cache" --include=*.py .
Whatever that is needs a config fingerprint in its key, or it will keep
serving answers generated under the old retrieval threshold.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("bgo_chatbot.cache")

SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
MAX_TURNS_PER_SESSION = int(os.getenv("MAX_TURNS_PER_SESSION", "20"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "2000"))
HISTORY_BACKEND = os.getenv("HISTORY_BACKEND", "file").lower()  # file | memory
HISTORY_PATH = Path(os.getenv("HISTORY_PATH", "data/processed/.chat_history.json"))

_lock = threading.RLock()
_chat_history: Dict[str, List[Tuple[str, str, datetime]]] = {}
_loaded = False


# ------------------------------------------------------------- persistence
def _serialize() -> dict:
    return {
        sid: [[q, a, ts.isoformat()] for q, a, ts in turns]
        for sid, turns in _chat_history.items()
    }


def _deserialize(raw: dict) -> Dict[str, List[Tuple[str, str, datetime]]]:
    out: Dict[str, List[Tuple[str, str, datetime]]] = {}
    for sid, turns in raw.items():
        parsed = []
        for t in turns:
            try:
                parsed.append((t[0], t[1], datetime.fromisoformat(t[2])))
            except (IndexError, ValueError, TypeError):
                continue
        if parsed:
            out[sid] = parsed
    return out


def _ensure_loaded() -> None:
    """Re-read from disk so sibling workers see each other's writes."""
    global _loaded
    if HISTORY_BACKEND != "file":
        _loaded = True
        return
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            _chat_history.clear()
            _chat_history.update(_deserialize(json.load(f)))
    except (OSError, json.JSONDecodeError):
        pass
    _loaded = True


def _persist() -> None:
    if HISTORY_BACKEND != "file":
        return
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_serialize(), f, ensure_ascii=False)
        os.replace(tmp, HISTORY_PATH)  # atomic: no half-written file on crash
    except OSError as e:
        logger.warning("Could not persist chat history: %s", e)


# ------------------------------------------------------------- maintenance
def _cleanup_expired_sessions() -> None:
    """Caller must hold _lock."""
    now = datetime.now()
    cutoff = timedelta(hours=SESSION_EXPIRY_HOURS)

    for sid in [
        sid
        for sid, turns in _chat_history.items()
        if not turns or now - turns[-1][2] > cutoff
    ]:
        _chat_history.pop(sid, None)
        logger.debug("Removed expired session: %s", sid)

    if len(_chat_history) > MAX_SESSIONS:
        by_age = sorted(_chat_history.items(), key=lambda kv: kv[1][-1][2])
        for sid, _ in by_age[: len(_chat_history) - MAX_SESSIONS]:
            _chat_history.pop(sid, None)


# ------------------------------------------------------------- public API
def add_to_history(session_id: str, question: str, answer: str) -> None:
    """Append a question-answer pair to a session's history."""
    if not session_id or not question:
        return

    with _lock:
        _ensure_loaded()
        _cleanup_expired_sessions()

        turns = _chat_history.setdefault(session_id, [])

        # Idempotence guard: process_query is wrapped in @retry, so a partial
        # failure can re-enter this path with identical arguments.
        if turns and turns[-1][0] == question and turns[-1][1] == answer:
            return

        turns.append((question, answer, datetime.now()))
        _chat_history[session_id] = turns[-MAX_TURNS_PER_SESSION:]
        _persist()

        logger.debug(
            "History for session %s: %d turns", session_id, len(_chat_history[session_id])
        )


def get_history(session_id: Optional[str], max_turns: int = 5) -> str:
    """
    Formatted history for the rewrite prompt.

    Labels are Portuguese because rewrite/prompts.py is Portuguese - keeping
    them as 'Q:'/'A:' made the rewriter occasionally treat the block as an
    exam transcript rather than a conversation.
    """
    turns = get_history_list(session_id, max_turns=max_turns)
    if not turns:
        return ""
    parts = []
    for q, a in turns:
        parts.append(f"Usuário: {q}")
        parts.append(f"Assistente: {a}")
    return "\n".join(parts)


def get_history_list(
    session_id: Optional[str], max_turns: int = 5
) -> List[Tuple[str, str]]:
    """History as (question, answer) tuples, oldest first."""
    if not session_id:
        return []
    with _lock:
        _ensure_loaded()
        turns = _chat_history.get(session_id)
        if not turns:
            return []
        return [(q, a) for q, a, _ in turns[-max_turns:]]


def clear_history(session_id: str) -> None:
    """Drop one session."""
    with _lock:
        _ensure_loaded()
        if _chat_history.pop(session_id, None) is not None:
            _persist()
            logger.debug("Cleared history for session %s", session_id)


def get_session_count() -> int:
    """Number of live sessions."""
    with _lock:
        _ensure_loaded()
        _cleanup_expired_sessions()
        return len(_chat_history)


def clear_all() -> int:
    """Drop everything. Useful between eval runs so state doesn't leak."""
    with _lock:
        _ensure_loaded()
        n = len(_chat_history)
        _chat_history.clear()
        _persist()
        return n