# src/rag_pipeline/rewrite/rewrite_service.py
"""
Rewrite service: transforma a pergunta do usuário em uma versão
autossuficiente para melhorar recuperação no RAG.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .prompts import (
    get_query_rewrite_prompt,
    detect_language,
)

# LLM assíncrono 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Using cheaper model for rewriting

async def rewrite_query(question: str, chat_history: str = "") -> str:
    """
    Recebe a pergunta e (opcionalmente) histórico de conversa e retorna
    uma versão reformulada/autossuficiente.
    
    Args:
        question: Pergunta original do usuário
        chat_history: Histórico de conversa formatado (pode ser vazio)
        
    Returns:
        Pergunta reescrita, otimizada para busca semântica
    """
  # Detect language and get appropriate prompt
    language = detect_language(question)
    
    # Determine if we should use minimal prompt
    use_minimal = not chat_history or not chat_history.strip()
    
    # Get formatted prompt
    prompt_text = get_query_rewrite_prompt(
        question=question,
        chat_history=chat_history,
        language=language,
        use_minimal=use_minimal
    )
    # chamada assíncrona ao modelo
    response = await llm.ainvoke(prompt_text)

    # response.content contém o texto gerado
    return response.content.strip()
