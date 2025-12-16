"""
Chat history management for conversation sessions.

This module provides a simple in-memory storage for chat history.
For production, consider upgrading to Redis or a database.

Each session maintains a conversation history as a list of (question, answer) pairs.
"""

from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("bgo_chatbot.cache")

# In-memory storage: session_id -> List[Tuple[question, answer, timestamp]]
_chat_history: Dict[str, List[Tuple[str, str, datetime]]] = {}

# Session expiry time (default: 24 hours)
SESSION_EXPIRY_HOURS = 24


def _cleanup_expired_sessions():
    """Remove expired sessions from memory."""
    now = datetime.now()
    expired_sessions = []
    
    for session_id, history in _chat_history.items():
        if not history:
            expired_sessions.append(session_id)
            continue
            
        # Check if last interaction is expired
        last_interaction_time = history[-1][2]
        if now - last_interaction_time > timedelta(hours=SESSION_EXPIRY_HOURS):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del _chat_history[session_id]
        logger.debug(f"Removed expired session: {session_id}")


def add_to_history(session_id: str, question: str, answer: str) -> None:
    """
    Add a question-answer pair to the chat history for a session.
    
    Args:
        session_id: Unique identifier for the conversation session
        question: User's question
        answer: System's answer
    """
    if not session_id:
        return
    
    _cleanup_expired_sessions()
    
    if session_id not in _chat_history:
        _chat_history[session_id] = []
    
    _chat_history[session_id].append((question, answer, datetime.now()))
    logger.debug(f"Added to history for session {session_id}: {len(_chat_history[session_id])} messages")


def get_history(session_id: Optional[str], max_turns: int = 5) -> str:
    """
    Retrieve chat history for a session, formatted as a string.
    
    Args:
        session_id: Unique identifier for the conversation session
        max_turns: Maximum number of recent turns to include (default: 5)
        
    Returns:
        Formatted chat history string, or empty string if no history exists
    """
    if not session_id or session_id not in _chat_history:
        return ""
    
    _cleanup_expired_sessions()
    
    history = _chat_history[session_id]
    if not history:
        return ""
    
    # Get last N turns
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    
    # Format as Q/A pairs
    formatted_parts = []
    for question, answer, _ in recent_history:
        formatted_parts.append(f"Q: {question}")
        formatted_parts.append(f"A: {answer}")
    
    return "\n".join(formatted_parts)


def clear_history(session_id: str) -> None:
    """
    Clear chat history for a specific session.
    
    Args:
        session_id: Unique identifier for the conversation session
    """
    if session_id in _chat_history:
        del _chat_history[session_id]
        logger.debug(f"Cleared history for session {session_id}")


def get_history_list(session_id: Optional[str], max_turns: int = 5) -> List[Tuple[str, str]]:
    """
    Retrieve chat history as a list of (question, answer) tuples.
    
    Args:
        session_id: Unique identifier for the conversation session
        max_turns: Maximum number of recent turns to include
        
    Returns:
        List of (question, answer) tuples
    """
    if not session_id or session_id not in _chat_history:
        return []
    
    history = _chat_history[session_id]
    if not history:
        return []
    
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    return [(q, a) for q, a, _ in recent_history]


def get_session_count() -> int:
    """Get the number of active sessions."""
    _cleanup_expired_sessions()
    return len(_chat_history)

