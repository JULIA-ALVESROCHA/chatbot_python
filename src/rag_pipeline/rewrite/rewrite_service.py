# src/rag_pipeline/rewrite/rewrite_service.py
"""
Rewrite service: transforma a pergunta do usuário em uma versão
autossuficiente para melhorar recuperação no RAG.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .prompts import QUERY_REWRITE_TEMPLATE, QUERY_REWRITE_MINIMAL_TEMPLATE


# LLM assíncrono — escolha o modelo que preferir (custo x velocidade)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Using cheaper model for rewriting

rewrite_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template=QUERY_REWRITE_TEMPLATE
)

rewrite_prompt_minimal = PromptTemplate(
    input_variables=["question"],
    template=QUERY_REWRITE_MINIMAL_TEMPLATE
)


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
    # Use minimal template if no history, full template if history exists
    if chat_history and chat_history.strip():
        prompt_text = rewrite_prompt.format(
            question=question,
            chat_history=chat_history
        )
    else:
        prompt_text = rewrite_prompt_minimal.format(question=question)

    # chamada assíncrona ao modelo
    response = await llm.ainvoke(prompt_text)

    # response.content contém o texto gerado
    return response.content.strip()
