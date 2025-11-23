# src/rag_pipeline/rewrite/rewrite_service.py
"""
Rewrite service: transforma a pergunta do usuário em uma versão
autossuficiente para melhorar recuperação no RAG.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# LLM assíncrono — escolha o modelo que preferir (custo x velocidade)
llm = ChatOpenAI(model="gpt-4", temperature=0)

rewrite_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
Reescreva a pergunta abaixo para que ela seja completamente autossuficiente,
removendo pronomes ambíguos, resolvendo contexto e deixando a pergunta clara
para um sistema de recuperação de documentos (RAG).

Chat History:
{chat_history}

Pergunta original:
{question}

Pergunta reescrita:
"""
)


async def rewrite_query(question: str, chat_history: str = "") -> str:
    """
    Recebe a pergunta e (opcionalmente) histórico de conversa e retorna
    uma versão reformulada/autossuficiente.
    """
    prompt_text = rewrite_prompt.format(
        question=question,
        chat_history=chat_history
    )

    # chamada assíncrona ao modelo
    response = await llm.ainvoke(prompt_text)

    # response.content contém o texto gerado
    return response.content.strip()
