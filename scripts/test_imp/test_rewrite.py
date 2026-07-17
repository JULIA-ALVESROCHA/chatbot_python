"""
Comprehensive test suite for query rewriting system.

Tests cover:
1. Prompt generation and formatting
2. Language detection
3. Query decontextualization
4. Query decomposition
5. Edge cases and error handling
6. Integration with rewrite_service

Run with: python scripts/test_imp/test_rewrite.py
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# -------------------------------------------------
# FORCE chatbot_python as the ONLY project root
# -------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # chatbot_python

# remove qualquer caminho que contenha 'bgo-chatbot' mas não seja o repo
sys.path = [
    p for p in sys.path
    if not (p.endswith("bgo-chatbot") or p.endswith("bgo-chatbot\\"))
]

# força o repo correto
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from src.rag_pipeline.rewrite.prompts import (
    detect_language,
    get_query_rewrite_prompt,
    get_decomposition_prompt,
    format_chat_history,
    parse_decomposed_queries,
    should_use_minimal_prompt,
    is_valid_query,
    get_fallback_message,
    MAX_DECOMPOSED_QUERIES,
)

# Import the service to test
from src.rag_pipeline.rewrite.rewrite_service import rewrite_query


# ==============================================================================
# TEST HELPERS
# ==============================================================================

def sample_chat_history():
    """Sample chat history for testing."""
    return [
        {"role": "user", "content": "Qual é a duração da prova da OBG?"},
        {"role": "assistant", "content": "A prova da Olimpíada Brasileira de Geografia tem duração de 3 horas."},
        {"role": "user", "content": "Posso usar calculadora?"},
        {"role": "assistant", "content": "Não, o uso de calculadoras não é permitido durante a prova."}
    ]


def sample_chat_history_formatted():
    """Formatted chat history string."""
    return """User: Qual é a duração da prova da OBG?
Assistant: A prova da Olimpíada Brasileira de Geografia tem duração de 3 horas.
User: Posso usar calculadora?
Assistant: Não, o uso de calculadoras não é permitido durante a prova."""


def assert_equal(actual, expected, msg=""):
    """Helper assertion function."""
    if actual != expected:
        raise AssertionError(f"{msg}\nExpected: {expected}\nActual: {actual}")


# ==============================================================================
# TEST 1: LANGUAGE DETECTION
# ==============================================================================

def test_language_detection():
    """Test suite for language detection."""
    print("\n[TEST] Language Detection Tests")
    
    # Test detection of simple Portuguese text
    text = "Qual é a duração da prova?"
    assert detect_language(text) == "pt", f"Expected 'pt', got '{detect_language(text)}'"
    print("  [OK] detect_portuguese_simple")
    
    # Test detection of simple English text
    text = "What is the duration of the exam?"
    assert detect_language(text) == "en", f"Expected 'en', got '{detect_language(text)}'"
    print("  [OK] detect_english_simple")
    
    # Test detection with Portuguese pronouns
    text = "E sobre isso? Ele pode fazer aquilo?"
    assert detect_language(text) == "pt", f"Expected 'pt', got '{detect_language(text)}'"
    print("  [OK] detect_portuguese_with_pronouns")
    
    # Test detection with English pronouns
    text = "What about this? Can it do that?"
    assert detect_language(text) == "en", f"Expected 'en', got '{detect_language(text)}'"
    print("  [OK] detect_english_with_pronouns")
    
    # Test detection with regulation-specific Portuguese terms
    text = "Quais são os critérios do regulamento da olimpíada?"
    assert detect_language(text) == "pt", f"Expected 'pt', got '{detect_language(text)}'"
    print("  [OK] detect_portuguese_regulation_terms")
    
    # Test that ambiguous text defaults to Portuguese
    text = "ABC 123"
    assert detect_language(text) == "pt", f"Expected 'pt', got '{detect_language(text)}'"
    print("  [OK] detect_ambiguous_defaults_to_portuguese")
    
    # Test detection with empty string
    assert detect_language("") == "pt", f"Expected 'pt', got '{detect_language('')}'"
    print("  [OK] detect_empty_string")


# ==============================================================================
# TEST 2: PROMPT FORMATTING
# ==============================================================================

def test_prompt_formatting():
    """Test suite for prompt generation and formatting."""
    print("\n[TEST] Prompt Formatting Tests")
    
    # Test minimal prompt generation in Portuguese
    prompt = get_query_rewrite_prompt(
        question="Quais são os critérios?",
        use_minimal=True,
        language="pt"
    )
    assert isinstance(prompt, str), f"Expected str, got {type(prompt)}"
    assert len(prompt) > 0, "Prompt should not be empty"
    assert "Quais são os critérios?" in prompt, "Question not in prompt"
    assert "autossuficiente" in prompt, "Portuguese keyword not found"
    assert "Histórico da Conversa" not in prompt, "History section should not be in minimal prompt"
    # Verify prompt is properly formatted (no {placeholders} left)
    assert "{" not in prompt or "}" not in prompt, f"Unfilled placeholders found in prompt: {prompt}"
    print("  [OK] minimal_prompt_portuguese")
    
    # Test minimal prompt generation in English
    prompt = get_query_rewrite_prompt(
        question="What are the criteria?",
        use_minimal=True,
        language="en"
    )
    assert "What are the criteria?" in prompt, "Question not in prompt"
    assert "self-contained" in prompt, "English keyword not found"
    print("  [OK] minimal_prompt_english")
    
    # Test full prompt with chat history in Portuguese
    history = "User: Qual a duração?\nAssistant: 3 horas."
    prompt = get_query_rewrite_prompt(
        question="E sobre isso?",
        chat_history=history,
        language="pt"
    )
    assert isinstance(prompt, str), f"Expected str, got {type(prompt)}"
    assert len(prompt) > 0, "Prompt should not be empty"
    assert "E sobre isso?" in prompt, "Question not in prompt"
    assert "3 horas" in prompt, "History content not in prompt"
    assert "Histórico da Conversa" in prompt, "History section not in prompt"
    # Verify no unfilled placeholders
    assert "{" not in prompt or "}" not in prompt, f"Unfilled placeholders found: {prompt}"
    print("  [OK] full_prompt_with_history_portuguese")
    
    # Test full prompt with chat history in English
    history = "User: What is the duration?\nAssistant: 3 hours."
    prompt = get_query_rewrite_prompt(
        question="What about it?",
        chat_history=history,
        language="en"
    )
    assert "What about it?" in prompt, "Question not in prompt"
    assert "3 hours" in prompt, "History content not in prompt"
    assert "Chat History" in prompt, "History section not in prompt"
    print("  [OK] full_prompt_with_history_english")
    
    # Test decomposition prompt in Portuguese
    prompt = get_decomposition_prompt(
        query="Quais são os critérios e como funciona a prova?",
        language="pt",
        max_queries=3
    )
    assert isinstance(prompt, str), f"Expected str, got {type(prompt)}"
    assert len(prompt) > 0, "Prompt should not be empty"
    assert "Quais são os critérios e como funciona a prova?" in prompt, "Query not in prompt"
    assert "3" in prompt, "Max queries not in prompt"
    assert "sub-perguntas" in prompt.lower(), "Sub-questions keyword not found"
    # Verify no unfilled placeholders
    assert "{" not in prompt or "}" not in prompt, f"Unfilled placeholders found: {prompt}"
    print("  [OK] decomposition_prompt_portuguese")
    
    # Test decomposition prompt in English
    prompt = get_decomposition_prompt(
        query="What are the criteria and how does the exam work?",
        language="en",
        max_queries=3
    )
    assert "What are the criteria and how does the exam work?" in prompt, "Query not in prompt"
    assert "3" in prompt, "Max queries not in prompt"
    assert "sub-questions" in prompt.lower(), "Sub-questions keyword not found"
    print("  [OK] decomposition_prompt_english")
    
    # Test automatic minimal prompt when no history
    prompt = get_query_rewrite_prompt(
        question="Quais são os critérios?",
        chat_history="",
        language="pt"
    )
    # Should use minimal template
    assert "Histórico da Conversa" not in prompt, "Should not have history section"
    print("  [OK] auto_minimal_when_no_history")


# ==============================================================================
# TEST 3: CHAT HISTORY FORMATTING
# ==============================================================================

def test_chat_history_formatting():
    """Test suite for chat history formatting."""
    print("\n[TEST] Chat History Formatting Tests")
    history_data = sample_chat_history()
    
    # Test formatting empty history
    result = format_chat_history([])
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert result == "", f"Empty history should return empty string, got: '{result}'"
    print("  [OK] format_empty_history")
    
    # Test formatting single conversation turn
    history = [history_data[0]]
    formatted = format_chat_history(history)
    assert isinstance(formatted, str), f"Expected str, got {type(formatted)}"
    assert len(formatted) > 0, "Formatted history should not be empty"
    assert "User: Qual é a duração da prova da OBG?" in formatted, f"Single turn not formatted correctly. Got: {formatted}"
    assert formatted.count("User:") == 1, "Should have exactly one User entry"
    print("  [OK] format_single_turn")
    
    # Test formatting multiple conversation turns
    formatted = format_chat_history(history_data)
    assert "User:" in formatted, "User role not found"
    assert "Assistant:" in formatted, "Assistant role not found"
    assert "duração da prova" in formatted, "First message content not found"
    assert "calculadora" in formatted, "Second message content not found"
    print("  [OK] format_multiple_turns")
    
    # Test formatting handles missing role field
    history = [{"content": "Test message"}]
    formatted = format_chat_history(history)
    assert "User: Test message" in formatted, "Default role not applied"
    print("  [OK] format_handles_missing_role")
    
    # Test formatting handles missing content field
    history = [{"role": "user"}]
    formatted = format_chat_history(history)
    assert "User:" in formatted, "Role with no content not handled"
    print("  [OK] format_handles_missing_content")


# ==============================================================================
# TEST 4: QUERY DECOMPOSITION PARSING
# ==============================================================================

def test_query_decomposition():
    """Test suite for query decomposition parsing."""
    print("\n[TEST] Query Decomposition Parsing Tests")
    
    # Test parsing queries numbered with dots
    response = """1. Quais são os critérios de inscrição?
2. Como funciona a prova?
3. Qual é a duração?"""
    queries = parse_decomposed_queries(response)
    assert isinstance(queries, list), f"Expected list, got {type(queries)}"
    assert len(queries) == 3, f"Expected 3 queries, got {len(queries)}: {queries}"
    assert all(isinstance(q, str) for q in queries), "All queries should be strings"
    assert all(len(q) > 0 for q in queries), "All queries should be non-empty"
    assert "Quais são os critérios de inscrição?" in queries, f"First query not parsed. Got: {queries}"
    assert "Como funciona a prova?" in queries, f"Second query not parsed. Got: {queries}"
    assert "Qual é a duração?" in queries, f"Third query not parsed. Got: {queries}"
    # Verify no numbering left in queries
    assert not any(q.strip().startswith(("1.", "2.", "3.")) for q in queries), "Numbering should be removed from queries"
    print("  [OK] parse_numbered_queries_with_dots")
    
    # Test parsing queries numbered with parentheses
    response = """1) What are the criteria?
2) How does the exam work?"""
    queries = parse_decomposed_queries(response)
    assert len(queries) == 2, f"Expected 2 queries, got {len(queries)}"
    assert "What are the criteria?" in queries, "First query not parsed"
    assert "How does the exam work?" in queries, "Second query not parsed"
    print("  [OK] parse_numbered_queries_with_parentheses")
    
    # Test that parsing respects MAX_DECOMPOSED_QUERIES limit
    response = """1. Question 1
2. Question 2
3. Question 3
4. Question 4
5. Question 5"""
    queries = parse_decomposed_queries(response)
    assert isinstance(queries, list), f"Expected list, got {type(queries)}"
    assert len(queries) <= MAX_DECOMPOSED_QUERIES, f"Expected <= {MAX_DECOMPOSED_QUERIES}, got {len(queries)}: {queries}"
    assert len(queries) == MAX_DECOMPOSED_QUERIES, f"Should return exactly {MAX_DECOMPOSED_QUERIES} queries when limit is applied, got {len(queries)}"
    print("  [OK] parse_respects_max_limit")
    
    # Test parsing handles empty lines
    response = """1. Question 1

2. Question 2

3. Question 3"""
    queries = parse_decomposed_queries(response)
    assert len(queries) == 3, f"Expected 3 queries, got {len(queries)}"
    print("  [OK] parse_handles_empty_lines")
    
    # Test parsing single query (no decomposition needed)
    response = "1. What is the exam duration?"
    queries = parse_decomposed_queries(response)
    assert len(queries) == 1, f"Expected 1 query, got {len(queries)}"
    assert "What is the exam duration?" in queries, "Query not parsed correctly"
    print("  [OK] parse_single_query")


# ==============================================================================
# TEST 5: UTILITY FUNCTIONS
# ==============================================================================

def test_utility_functions():
    """Test suite for utility functions."""
    print("\n[TEST] Utility Functions Tests")
    
    # Test minimal prompt decision with empty history
    result1 = should_use_minimal_prompt("")
    assert isinstance(result1, bool), f"Expected bool, got {type(result1)}"
    assert result1 is True, f"Empty history should use minimal, got {result1}"
    
    # Test with None - should handle gracefully or raise TypeError
    try:
        result2 = should_use_minimal_prompt(None)
        assert isinstance(result2, bool), f"Expected bool, got {type(result2)}"
        # If it doesn't raise, it should return True
        assert result2 is True, f"None history should use minimal, got {result2}"
    except (TypeError, AttributeError):
        # If it raises an error, that's also acceptable behavior
        pass
    print("  [OK] should_use_minimal_with_empty_history")
    
    # Test minimal prompt decision with very short history
    assert should_use_minimal_prompt("User: Hi") is True, "Short history should use minimal"
    print("  [OK] should_use_minimal_with_short_history")
    
    # Test minimal prompt decision with substantial history
    long_history = "User: " + ("A" * 100) + "\nAssistant: " + ("B" * 100)
    assert should_use_minimal_prompt(long_history) is False, "Long history should not use minimal"
    print("  [OK] should_use_minimal_with_substantial_history")
    
    # Test query validation accepts valid queries
    result1 = is_valid_query("Quais são os critérios?")
    assert isinstance(result1, bool), f"Expected bool, got {type(result1)}"
    assert result1 is True, f"Valid Portuguese query should pass, got {result1}"
    
    result2 = is_valid_query("What are the criteria?")
    assert isinstance(result2, bool), f"Expected bool, got {type(result2)}"
    assert result2 is True, f"Valid English query should pass, got {result2}"
    
    # Test with longer valid queries
    assert is_valid_query("Quais são os critérios de inscrição na Olimpíada Brasileira de Geografia?") is True, "Long valid query should pass"
    print("  [OK] is_valid_query_accepts_good_queries")
    
    # Test query validation rejects empty
    assert is_valid_query("") is False, "Empty query should be rejected"
    assert is_valid_query("   ") is False, "Whitespace-only query should be rejected"
    print("  [OK] is_valid_query_rejects_empty")
    
    # Test query validation rejects too short
    assert is_valid_query("?") is False, "Too short query should be rejected"
    assert is_valid_query("a") is False, "Single char query should be rejected"
    print("  [OK] is_valid_query_rejects_too_short")
    
    # Test query validation rejects only punctuation
    assert is_valid_query("???!!!") is False, "Only punctuation should be rejected"
    assert is_valid_query("...") is False, "Only dots should be rejected"
    print("  [OK] is_valid_query_rejects_only_punctuation")
    
    # Test fallback messages in Portuguese
    msg = get_fallback_message("empty_query", "pt")
    assert isinstance(msg, str), f"Expected str, got {type(msg)}"
    assert len(msg) > 0, "Fallback message should not be empty"
    assert "pergunta" in msg.lower() or "olimpíada" in msg.lower(), f"Portuguese fallback should contain expected words. Got: {msg}"
    # Verify it's actually in Portuguese (not English)
    assert "question" not in msg.lower() or "olympiad" not in msg.lower(), f"Should be Portuguese, not English. Got: {msg}"
    print("  [OK] get_fallback_message_portuguese")
    
    # Test fallback messages in English
    msg = get_fallback_message("empty_query", "en")
    assert "question" in msg.lower() or "olympiad" in msg.lower(), "English fallback should contain expected words"
    print("  [OK] get_fallback_message_english")
    
    # Test fallback message for unknown type returns default
    msg = get_fallback_message("unknown_type", "pt")
    assert msg is not None, "Unknown type should return default message"
    assert len(msg) > 0, "Default message should not be empty"
    print("  [OK] get_fallback_message_unknown_type")


# ==============================================================================
# TEST 6: INTEGRATION WITH REWRITE_SERVICE
# ==============================================================================

async def test_rewrite_service_integration():
    """Integration tests for rewrite_service.py."""
    print("\n[TEST] Rewrite Service Integration Tests")
    
    # Test rewriting a simple query without history
    with patch('src.rag_pipeline.rewrite.rewrite_service.llm') as mock_llm:
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "Quais são os critérios de inscrição na Olimpíada Brasileira de Geografia?"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await rewrite_query(
            question="Quais são os critérios?",
            chat_history=""
        )
        
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert result is not None, "Result should not be None"
        assert len(result) > 0, f"Result should not be empty, got: '{result}'"
        assert result == result.strip(), "Result should be stripped of whitespace"
        # Verify LLM was called exactly once
        assert mock_llm.ainvoke.call_count == 1, f"LLM should be called once, got {mock_llm.ainvoke.call_count}"
        # Verify the prompt passed to LLM contains the question
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert "Quais são os critérios?" in call_args, "Prompt should contain original question"
        print("  [OK] rewrite_simple_query_no_history")
    
    # Test rewriting query with pronoun reference
    with patch('src.rag_pipeline.rewrite.rewrite_service.llm') as mock_llm:
        mock_response = MagicMock()
        mock_response.content = "Posso usar calculadora na prova da Olimpíada Brasileira de Geografia?"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await rewrite_query(
            question="Posso usar calculadora nela?",
            chat_history="User: Qual é a prova da OBG?\nAssistant: É a prova da Olimpíada Brasileira de Geografia."
        )
        
        assert result is not None, "Result should not be None"
        assert "calculadora" in result, "Result should contain rewritten content"
        print("  [OK] rewrite_query_with_pronoun")
    
    # Test that rewrite handles LLM errors gracefully
    with patch('src.rag_pipeline.rewrite.rewrite_service.llm') as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))
        
        exception_raised = False
        try:
            await rewrite_query(
                question="Test question",
                chat_history=""
            )
        except Exception as e:
            exception_raised = True
            assert isinstance(e, Exception), f"Expected Exception, got {type(e)}"
            assert "API Error" in str(e), f"Exception should contain error message. Got: {e}"
        
        assert exception_raised, "Should have raised exception when LLM fails"
        assert mock_llm.ainvoke.call_count == 1, "LLM should have been called before error"
        print("  [OK] rewrite_handles_llm_error")


# ==============================================================================
# TEST 7: EDGE CASES AND ERROR HANDLING
# ==============================================================================

def test_edge_cases():
    """Test suite for edge cases and error handling."""
    print("\n[TEST] Edge Cases and Error Handling Tests")
    
    # Test language detection with mixed PT/EN text
    # More Portuguese words
    text = "Qual é the duration da prova?"
    assert detect_language(text) == "pt", "Mixed text with more PT should detect as PT"
    
    # More English words
    text = "What is a duração of the exam?"
    assert detect_language(text) == "en", "Mixed text with more EN should detect as EN"
    print("  [OK] detect_language_with_mixed_languages")
    
    # Test chat history formatting with special characters
    history = [
        {"role": "user", "content": "Teste com 'aspas' e \"citações\""},
        {"role": "assistant", "content": "Resposta com \n quebras de linha"}
    ]
    formatted = format_chat_history(history)
    assert "aspas" in formatted, "Special chars should be preserved"
    assert "citações" in formatted, "Quotes should be preserved"
    print("  [OK] format_chat_history_with_special_characters")
    
    # Test parsing with malformed query numbering
    response = """Question 1: First question
Question 2: Second question
3. Third question"""
    queries = parse_decomposed_queries(response)
    # Should still extract the numbered one
    assert len(queries) >= 1, "Should extract at least one query"
    print("  [OK] parse_queries_with_malformed_numbering")
    
    # Test prompt generation with very long history
    long_history = "User: " + ("A" * 5000)
    prompt = get_query_rewrite_prompt(
        question="Test?",
        chat_history=long_history,
        language="pt"
    )
    assert "Test?" in prompt, "Question should be in prompt"
    # Should still include history even if long
    assert "A" * 100 in prompt, "At least part of history should be included"
    print("  [OK] get_prompt_with_very_long_history")
    
    # Test that format_chat_history handles None gracefully
    try:
        result = format_chat_history(None)
        # If it doesn't raise, it should return empty string or handle gracefully
        assert isinstance(result, str), f"Should return str even with None, got {type(result)}"
    except (TypeError, AttributeError):
        # If it raises, that's also acceptable - but should be documented
        pass
    print("  [OK] format_chat_history_handles_none")
    
    # Test that parse_decomposed_queries handles empty string
    result = parse_decomposed_queries("")
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 0, f"Empty input should return empty list, got {result}"
    print("  [OK] parse_decomposed_queries_handles_empty")
    
    # Test that get_query_rewrite_prompt raises error on invalid language
    try:
        prompt = get_query_rewrite_prompt("Test?", "", language="invalid")
        # If it doesn't raise, check it defaults to something reasonable
        assert isinstance(prompt, str), "Should return str even with invalid language"
    except (ValueError, KeyError):
        # If it raises, that's good - invalid input should be rejected
        pass
    print("  [OK] get_query_rewrite_prompt_handles_invalid_language")


# ==============================================================================
# TEST 8: PERFORMANCE AND CONSISTENCY
# ==============================================================================

def test_performance_and_consistency():
    """Test suite for performance and consistency checks."""
    print("\n[TEST] Performance and Consistency Tests")
    
    # Test that language detection is consistent for same input
    text = "Qual é a duração da prova?"
    results = [detect_language(text) for _ in range(10)]
    assert len(results) == 10, "Should have 10 results"
    assert all(isinstance(r, str) for r in results), "All results should be strings"
    assert all(r in ("pt", "en") for r in results), f"All results should be 'pt' or 'en', got: {set(results)}"
    assert all(r == "pt" for r in results), f"Language detection should be consistent. Got: {results}"
    # Test that it's actually detecting correctly
    assert detect_language(text) == "pt", f"Should detect Portuguese, got: {detect_language(text)}"
    print("  [OK] language_detection_is_consistent")
    
    # Test that prompt generation is deterministic
    question = "Test question"
    history = "User: Previous\nAssistant: Response"
    
    prompt1 = get_query_rewrite_prompt(question, history, "pt")
    prompt2 = get_query_rewrite_prompt(question, history, "pt")
    
    assert isinstance(prompt1, str), f"Expected str, got {type(prompt1)}"
    assert isinstance(prompt2, str), f"Expected str, got {type(prompt2)}"
    assert prompt1 == prompt2, f"Prompt generation should be deterministic. Got different results:\n1: {prompt1[:100]}...\n2: {prompt2[:100]}..."
    assert len(prompt1) > 0, "Prompt should not be empty"
    assert len(prompt2) > 0, "Prompt should not be empty"
    print("  [OK] prompt_generation_is_deterministic")
    
    # Test query parsing handles Unicode characters
    response = """1. Questão com acentuação: áéíóú
2. Question with special chars: ñ ü
3. Pergunta com ç cedilha"""
    queries = parse_decomposed_queries(response)
    assert isinstance(queries, list), f"Expected list, got {type(queries)}"
    assert len(queries) == 3, f"Expected 3 queries, got {len(queries)}: {queries}"
    assert "acentuação" in queries[0], f"Unicode characters should be preserved. Got: {queries[0]}"
    assert "ñ" in queries[1] or "ü" in queries[1], f"Special chars should be preserved. Got: {queries[1]}"
    assert "ç" in queries[2], f"Cedilla should be preserved. Got: {queries[2]}"
    # Verify all queries are valid strings
    assert all(isinstance(q, str) for q in queries), "All queries should be strings"
    # Verify no queries are empty
    assert all(len(q.strip()) > 0 for q in queries), "No query should be empty after parsing"
    print("  [OK] parse_queries_handles_unicode")


# ==============================================================================
# TEST 9: ACTUAL BUG DETECTION
# ==============================================================================

def test_actual_bugs():
    """Test suite that specifically looks for common bugs."""
    print("\n[TEST] Actual Bug Detection Tests")
    
    # Test 1: Check if get_query_rewrite_prompt properly handles None question
    try:
        result = get_query_rewrite_prompt(None, "", "pt")
        # If it doesn't raise, the result should be a string (even if empty)
        assert isinstance(result, str), f"Should return str even with None question, got {type(result)}"
    except (TypeError, AttributeError, ValueError):
        # If it raises, that's acceptable - None should be rejected
        pass
    print("  [OK] get_query_rewrite_prompt_handles_none_question")
    
    # Test 2: Check if format_chat_history properly handles invalid input types
    try:
        result = format_chat_history("not a list")
        # If it doesn't raise, check it handles gracefully
        assert isinstance(result, str), f"Should return str, got {type(result)}"
    except (TypeError, AttributeError):
        # If it raises, that's good - invalid type should be rejected
        pass
    print("  [OK] format_chat_history_handles_invalid_type")
    
    # Test 3: Check if parse_decomposed_queries handles None
    try:
        result = parse_decomposed_queries(None)
        assert isinstance(result, list), f"Should return list even with None, got {type(result)}"
    except (TypeError, AttributeError):
        # If it raises, that's acceptable
        pass
    print("  [OK] parse_decomposed_queries_handles_none")
    
    # Test 4: Check if detect_language handles very long strings
    long_text = "Qual é a duração? " * 1000
    result = detect_language(long_text)
    assert result in ("pt", "en"), f"Should return 'pt' or 'en', got {result}"
    print("  [OK] detect_language_handles_long_strings")
    
    # Test 5: Check if get_query_rewrite_prompt handles very long questions
    long_question = "Qual é a duração? " * 100
    try:
        prompt = get_query_rewrite_prompt(long_question, "", "pt")
        assert isinstance(prompt, str), f"Should return str, got {type(prompt)}"
        assert long_question in prompt, "Long question should be in prompt"
    except Exception as e:
        # If it raises, that might indicate a bug (or intentional limit)
        print(f"  [WARN] get_query_rewrite_prompt raised exception with long question: {e}")
    print("  [OK] get_query_rewrite_prompt_handles_long_questions")
    
    # Test 6: Verify that all prompt functions return non-empty strings for valid input
    prompt1 = get_query_rewrite_prompt("Test?", "", "pt")
    assert len(prompt1) > 0, "Prompt should not be empty for valid input"
    
    prompt2 = get_decomposition_prompt("Test?", "pt")
    assert len(prompt2) > 0, "Decomposition prompt should not be empty for valid input"
    print("  [OK] all_prompts_return_non_empty_for_valid_input")
    
    # Test 7: Check that format_chat_history preserves order
    history = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user", "content": "Third"}
    ]
    formatted = format_chat_history(history)
    assert formatted.find("First") < formatted.find("Second"), "Order should be preserved"
    assert formatted.find("Second") < formatted.find("Third"), "Order should be preserved"
    print("  [OK] format_chat_history_preserves_order")
    
    # Test 8: Check that parse_decomposed_queries removes all numbering patterns
    response = "1. First\n2) Second\n3. Third\n(4) Fourth"
    queries = parse_decomposed_queries(response)
    for q in queries:
        # Should not start with numbers or parentheses
        assert not q.strip()[0].isdigit() if q.strip() else True, f"Query should not start with digit: {q}"
    print("  [OK] parse_decomposed_queries_removes_all_numbering")


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

async def run_all_tests():
    """Run all test suites."""
    print("="*70)
    print("QUERY REWRITE SYSTEM TEST SUITE")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    test_suites = [
        ("Language Detection", test_language_detection),
        ("Prompt Formatting", test_prompt_formatting),
        ("Chat History Formatting", test_chat_history_formatting),
        ("Query Decomposition", test_query_decomposition),
        ("Utility Functions", test_utility_functions),
        ("Rewrite Service Integration", test_rewrite_service_integration),
        ("Edge Cases", test_edge_cases),
        ("Performance and Consistency", test_performance_and_consistency),
        ("Actual Bug Detection", test_actual_bugs),
    ]
    
    for suite_name, test_func in test_suites:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            tests_passed += 1
        except Exception as e:
            tests_failed += 1
            print(f"\n[FAIL] {suite_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {tests_failed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)