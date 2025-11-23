# src/rag_pipeline/generator/answer_service.py
"""
Answer service: a partir da pergunta e dos documentos recuperados,
gera a resposta final usando o LLM e retorna também as fontes.
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# LLM para geração final — mantenha temperatura baixa para respostas factuais
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

answer_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Você é um assistente que responde perguntas com base exclusivamente nos trechos fornecidos abaixo.
Use apenas o conteúdo fornecido no campo "Contexto" para responder — não invente informações.
Se possível, cite a fonte de onde a informação veio (campo "source" do trecho).

Contexto (trechos extraídos do regulamento):
{context}

Pergunta:
{question}

Instruções:
- Responda em português (mesmo que a pergunta venha em outro idioma).
- Seja direto e objetivo.
- Ao final, inclua uma seção "Fontes" listando as fontes (metadata) usadas.
"""
)


async def generate_answer(question: str, docs: List[Any]) -> Dict[str, Any]:
    """
    Gera a resposta final com base na pergunta e na lista de Document objects (docs).
    Cada doc é esperado ter `page_content` e `metadata` (p.ex. {'source': 'regulamento.pdf', 'page': 3}).
    Retorna um dicionário: {"answer": <str>, "sources": <list>}
    """

    # Monta o contexto: cada trecho com um cabeçalho contendo a metadata relevante
    context_parts = []
    sources = []
    for d in docs:
        content = getattr(d, "page_content", str(d))
        metadata = getattr(d, "metadata", {}) or {}
        # padroniza fonte
        src = metadata.get("source") or metadata.get("file") or metadata.get("path") or "unknown"
        page = metadata.get("page") or metadata.get("page_number")
        # trecho curto para referência
        snippet = content.strip().replace("\n", " ")[:1000]
        # acumula contexto para o prompt
        context_parts.append(f"[source: {src} | page: {page}]\n{snippet}\n")
        # salva metadados resumidos para a resposta
        sources.append({"source": src, "page": page, "snippet": snippet})

    context_text = "\n\n".join(context_parts)

    prompt_text = answer_template.format(question=question, context=context_text)

    # Chamada assíncrona ao LLM
    response = await llm.ainvoke(prompt_text)
    answer_text = response.content.strip()

    # Retorna estrutura consistente
    return {"answer": answer_text, "sources": sources}
