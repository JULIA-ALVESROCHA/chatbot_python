import sys
import asyncio
from pathlib import Path
from typing import List

# -------------------------------------------------
# FORCE chatbot_python as the ONLY project root
# -------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
# File structure: chatbot_python/scripts/test_imp/test_generator_components.py
# parents[0] = scripts/test_imp/
# parents[1] = scripts/
# parents[2] = chatbot_python/ (PROJECT_ROOT)
PROJECT_ROOT = CURRENT_FILE.parents[2]  # chatbot_python

# remove qualquer caminho que contenha 'bgo-chatbot' mas não seja o repo
sys.path = [
    p for p in sys.path
    if not (p.endswith("bgo-chatbot") or p.endswith("bgo-chatbot\\"))
]

# força o repo correto
sys.path.insert(0, str(PROJECT_ROOT))

# sanity check - handle case where src might be namespace package (no __file__)
import src
if hasattr(src, '__file__') and src.__file__:
    print("[OK] SRC being used from:", src.__file__)
else:
    # Check if we can import from src successfully
    try:
        from src.app.core import config
        config_path = Path(config.__file__).parent.parent.parent
        print("[OK] SRC package location (via config import):", config_path)
        print("[OK] SRC is a namespace package (no __file__ attribute)")
    except ImportError as e:
        print(f"[ERROR] Failed to import from src: {e}")
        print(f"[ERROR] PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"[ERROR] sys.path: {sys.path[:3]}")
        sys.exit(1)

# Now import the components to test
from langchain_core.documents import Document
from src.rag_pipeline.generator.answer_service import AnswerService
from src.rag_pipeline.generator.templates import (
    SYSTEM_PROMPT,
    ANSWER_TEMPLATE,
    FALLBACK_RESPONSE,
)


def test_templates_build_prompt():
    """
    Test that templates are properly defined and can be used to build prompts.
    """
    print("\n" + "="*60)
    print("TEST: test_templates_build_prompt")
    print("="*60)
    
    # Test 1: Check that all template constants exist
    assert SYSTEM_PROMPT is not None, "SYSTEM_PROMPT should not be None"
    assert ANSWER_TEMPLATE is not None, "ANSWER_TEMPLATE should not be None"
    assert FALLBACK_RESPONSE is not None, "FALLBACK_RESPONSE should not be None"
    print("[OK] All template constants are defined")
    
    # Test 2: Check that templates contain expected placeholders
    assert "{context}" in ANSWER_TEMPLATE, "ANSWER_TEMPLATE should contain {context} placeholder"
    assert "{question}" in ANSWER_TEMPLATE, "ANSWER_TEMPLATE should contain {question} placeholder"
    print("[OK] ANSWER_TEMPLATE contains required placeholders")
    
    # Test 3: Check that SYSTEM_PROMPT mentions OBG
    assert "Olimpíada Brasileira de Geografia" in SYSTEM_PROMPT or "OBG" in SYSTEM_PROMPT, \
        "SYSTEM_PROMPT should mention OBG"
    print("[OK] SYSTEM_PROMPT mentions OBG")
    
    # Test 4: Test prompt formatting
    test_context = "Test context content"
    test_question = "Test question?"
    formatted = ANSWER_TEMPLATE.format(context=test_context, question=test_question)
    assert test_context in formatted, "Formatted template should contain context"
    assert test_question in formatted, "Formatted template should contain question"
    print("[OK] Template formatting works correctly")
    
    # Test 5: Check FALLBACK_RESPONSE is a string
    assert isinstance(FALLBACK_RESPONSE, str), "FALLBACK_RESPONSE should be a string"
    assert len(FALLBACK_RESPONSE) > 0, "FALLBACK_RESPONSE should not be empty"
    print("[OK] FALLBACK_RESPONSE is a valid string")
    
    print("\n[PASS] test_templates_build_prompt - All checks passed!\n")
    return True


async def test_answer_service_with_documents():
    """
    Test AnswerService.generate_answer() with valid documents.
    """
    print("\n" + "="*60)
    print("TEST: test_answer_service_with_documents")
    print("="*60)
    
    # Create test documents
    test_docs = [
        Document(
            page_content=(
                "A Olimpíada Brasileira de Geografia é destinada a estudantes "
                "do ensino médio regularmente matriculados em escolas públicas ou privadas."
            ),
            metadata={"source": "regulamento.pdf", "page": 1},
        ),
        Document(
            page_content=(
                "A fase final da competição ocorre presencialmente "
                "no mês de dezembro em local a ser definido."
            ),
            metadata={"source": "regulamento.pdf", "page": 5},
        ),
    ]
    
    # Initialize service
    service = AnswerService(model_name="gpt-4o-mini")  # Use cheaper model for testing
    print("[OK] AnswerService initialized")
    
    # Test 1: Check that prompt is created
    assert service.prompt is not None, "Prompt template should be initialized"
    print("[OK] Prompt template initialized")
    
    # Test 2: Test _build_context method
    context = service._build_context(test_docs)
    assert isinstance(context, str), "Context should be a string"
    assert len(context) > 0, "Context should not be empty"
    assert test_docs[0].page_content in context, "Context should contain first document"
    assert test_docs[1].page_content in context, "Context should contain second document"
    print("[OK] _build_context() works correctly")
    print(f"   Context preview: {context[:100]}...")
    
    # Test 3: Generate answer with documents
    question = "Quem pode participar da Olimpíada Brasileira de Geografia?"
    try:
        answer = await service.generate_answer(question, test_docs)
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        assert answer != FALLBACK_RESPONSE, "Answer should not be fallback when documents are provided"
        print("[OK] generate_answer() returned valid answer")
        print(f"   Answer preview: {answer[:150]}...")
    except Exception as e:
        print(f"[ERROR] generate_answer() failed: {e}")
        raise
    
    print("\n[PASS] test_answer_service_with_documents - All checks passed!\n")
    return True


async def test_answer_service_without_documents():
    """
    Test AnswerService.generate_answer() with empty document list.
    Should return FALLBACK_RESPONSE.
    """
    print("\n" + "="*60)
    print("TEST: test_answer_service_without_documents")
    print("="*60)
    
    # Initialize service
    service = AnswerService(model_name="gpt-4o-mini")
    print("[OK] AnswerService initialized")
    
    # Test 1: Empty list should return fallback
    empty_docs: List[Document] = []
    question = "Qualquer pergunta?"
    
    try:
        answer = await service.generate_answer(question, empty_docs)
        assert answer == FALLBACK_RESPONSE, \
            f"Empty documents should return FALLBACK_RESPONSE, got: {answer}"
        print("[OK] Empty documents list returns FALLBACK_RESPONSE")
        print(f"   Fallback response: {answer}")
    except Exception as e:
        print(f"[ERROR] generate_answer() with empty docs failed: {e}")
        raise
    
    # Test 2: None should also return fallback (if handled)
    # Note: This might raise an error, which is acceptable behavior
    try:
        answer = await service.generate_answer(question, None)  # type: ignore
        if answer == FALLBACK_RESPONSE:
            print("[OK] None documents handled correctly")
        else:
            print("[WARN] None documents did not return FALLBACK_RESPONSE")
    except (TypeError, AttributeError):
        print("[OK] None documents raises appropriate error (acceptable behavior)")
    
    print("\n[PASS] test_answer_service_without_documents - All checks passed!\n")
    return True


async def main():
    """
    Run all tests and provide a summary report.
    """
    print("\n" + "="*60)
    print("GENERATOR COMPONENTS TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Templates
    try:
        results["test_templates_build_prompt"] = test_templates_build_prompt()
    except Exception as e:
        print(f"[FAIL] test_templates_build_prompt failed: {e}")
        results["test_templates_build_prompt"] = False
    
    # Test 2: AnswerService with documents
    try:
        results["test_answer_service_with_documents"] = await test_answer_service_with_documents()
    except Exception as e:
        print(f"[FAIL] test_answer_service_with_documents failed: {e}")
        results["test_answer_service_with_documents"] = False
    
    # Test 3: AnswerService without documents
    try:
        results["test_answer_service_without_documents"] = await test_answer_service_without_documents()
    except Exception as e:
        print(f"[FAIL] test_answer_service_without_documents failed: {e}")
        results["test_answer_service_without_documents"] = False
    
    # Summary report
    print("\n" + "="*60)
    print("TEST SUMMARY REPORT")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
