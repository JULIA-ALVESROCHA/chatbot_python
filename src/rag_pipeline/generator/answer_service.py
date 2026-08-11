"""
src/rag_pipeline/generator/answer_service.py

MUDANCAS
--------
1. max_tokens 300 -> 800. Era teto duplo junto com "2-3 frases": as
   respostas gold têm 5,08 fatos atômicos e não cabiam. Pior, enumerações
   eram cortadas no meio da frase e o decompositor transformava o fragmento
   em fato malformado, derrubando precisão E recall.

2. ChatPromptTemplate REMOVIDO. Ele trata toda {chave} como variável, e o
   SYSTEM_PROMPT novo tem {refusal_pt}/{refusal_en}, o ANSWER_TEMPLATE tem
   {calendario}. format_messages() levantaria KeyError. Usamos as funções
   answer_prompt()/system_prompt(), que fazem a injeção corretamente.

3. refused: bool no retorno. Hoje a avaliação não distingue "recusou" de
   "errou" — 39,5% das linhas são recusas e contaminam o FactScore.

4. _detect_phase MANTIDO. O rótulo "aplica-se a: <fase>" é real e o prompt
   depende dele. (Correção: eu havia dito que não existia.)

5. _extract_sources passa a preferir metadata["item"] ao regex sobre o texto.
   O splitter novo já grava o número do item; o regex pegava qualquer
   sequência numérica do chunk, inclusive datas e valores.
"""

from typing import Any, Dict, List, Optional
import logging
import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.app.core.config import settings
from src.rag_pipeline.generator.templates import (
    answer_prompt,
    system_prompt,
    is_refusal,
    refusal,
)

logger = logging.getLogger("bgo_chatbot.generator")


class AnswerService:
    """
    Gera a resposta final a partir dos documentos reranqueados.
    Nenhum texto de prompt vive aqui — tudo em templates.py.
    """

    def __init__(self, model_name: str = None, temperature: float = None):
        self.llm = ChatOpenAI(
            model=model_name or getattr(settings, "generation_model", "gpt-4o-mini"),
            temperature=(
                temperature
                if temperature is not None
                else getattr(settings, "generation_temperature", 0.1)
            ),
            request_timeout=30,
            # 800 comporta uma enumeração completa sem truncar. Se subir mais,
            # o modelo começa a divagar (é o problema oposto, documentado no
            # CanLegalRAGBench: respostas longas com material irrelevante).
            max_tokens=getattr(settings, "generation_max_tokens", 800),
        )

    async def generate_answer(
        self,
        question: str,
        documents: List[Document],
        language: str = "pt",
        chat_history: str = "",
    ) -> Dict[str, Any]:
        """
        Retorna:
            answer:   str
            sources:  List[Dict]
            refused:  bool   <- novo; use no eval e no cache
        """
        # Normaliza 'pt-BR' -> 'pt'. langdetect devolve 'pt'/'en'; o default
        # antigo era 'pt-BR' e o prompt só testava == 'en'.
        language = "en" if str(language).lower().startswith("en") else "pt"

        if not documents:
            # Com o piso min_chunks no _fuse_and_select isto virou caminho
            # raro, mas mantemos a sentinela por segurança.
            logger.warning("Nenhum documento recebido para geração")
            return {"answer": refusal(language), "sources": [], "refused": True}

        try:
            context = self._build_context_with_labels(documents)
            logger.debug("Gerando resposta com %d documentos", len(documents))

            messages = [
                {"role": "system", "content": system_prompt(language)},
                {"role": "user", "content": answer_prompt(
                    context=context,
                    question=question,
                    chat_history=chat_history,
                )},
            ]

            response = await self.llm.ainvoke(messages)
            answer_text = (response.content or "").strip()

            if not answer_text:
                logger.warning("LLM devolveu resposta vazia")
                return {"answer": refusal(language), "sources": [], "refused": True}

            answer_text = self._clean(answer_text)

            # Duas camadas: sentinela exata (barata e confiável) e heurística
            # de segurança para variações que o modelo insista em produzir.
            refused = is_refusal(answer_text) or self._looks_like_abstention(answer_text)

            if refused:
                # Citação sob "não sei" passa falsa autoridade ao usuário.
                sources = []
                # Normaliza qualquer variação para a sentinela canônica, para
                # que o avaliador consiga detectá-la por comparação exata.
                if not is_refusal(answer_text):
                    logger.info("Recusa fora do padrão normalizada: %r", answer_text[:80])
                    answer_text = refusal(language)
            else:
                sources = self._extract_sources(documents)

            logger.info(
                "Resposta gerada (len=%d, fontes=%d, recusa=%s)",
                len(answer_text), len(sources), refused,
            )
            return {"answer": answer_text, "sources": sources, "refused": refused}

        except Exception as e:
            logger.exception("Erro na geração da resposta: %s", e)
            raise

    # ------------------------------------------------------------ limpeza
    @staticmethod
    def _clean(text: str) -> str:
        """Remove prefixos e rodapés de citação que o modelo às vezes inventa."""
        text = re.sub(r"^\s*(Resposta|Response)\s*:\s*", "", text, flags=re.IGNORECASE)
        for pat in (
            r"\n\s*(você pode encontrar|encontre mais em|consulte|fonte|fontes|referência)s?\s*:.*$",
        ):
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    @staticmethod
    def _looks_like_abstention(text: str) -> bool:
        """
        Rede de segurança. A sentinela canônica é o caminho principal; isto
        pega variações. Mantido estreito de propósito: padrões largos como
        "não consta" pegariam respostas legítimas ("não consta limite de
        equipes por escola"), marcando conteúdo válido como recusa.
        """
        t = (text or "").lower()
        return any(p in t for p in (
            "não encontrei essa informação", "nao encontrei essa informacao",
            "não encontrei informações", "nao encontrei informacoes",
            "o contexto não menciona", "o contexto nao menciona",
            "não há informações nos documentos", "nao ha informacoes nos documentos",
            "os documentos não especificam", "os documentos nao especificam",
            "i could not find this information",
            "the context does not mention",
        ))

    # ------------------------------------------------------------ contexto
    @staticmethod
    def _detect_phase(text: str) -> str:
        """
        A qual fase da OBG o trecho se refere. Rotula o contexto para impedir
        que o gerador aplique regra de uma fase à outra (ex.: consulta a
        materiais é permitida online, não necessariamente na presencial).
        """
        t = (text or "").lower()
        has_presencial = "presencial" in t
        has_online = bool(re.search(r"\bonline\b|fases?\s+on-?line", t))
        if has_presencial and has_online:
            return "menciona fase online E fase presencial — atenção ao trecho exato"
        if has_presencial:
            return "fase presencial"
        if has_online:
            return "fases online"
        return "fase não especificada no trecho"

    @staticmethod
    def _build_context_with_labels(documents: List[Document]) -> str:
        parts = []
        for idx, doc in enumerate(documents, 1):
            meta = doc.metadata or {}
            source = str(meta.get("source", "Regulamento"))
            source_clean = re.sub(r"\.(pdf|txt)$", "", source.split("/")[-1],
                                  flags=re.IGNORECASE)
            page = meta.get("page", "?")
            item = meta.get("item")

            phase = AnswerService._detect_phase(doc.page_content)
            bits = [f"Fonte {idx}: {source_clean}-pag{page}"]
            if item:
                bits.append(f"item {item}")
            bits.append(f"aplica-se a: {phase}")

            parts.append(f"[{' | '.join(bits)}]\n{doc.page_content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------ fontes
    @staticmethod
    def _extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        seen = set()

        for doc in documents:
            meta = doc.metadata or {}
            key = (meta.get("source", ""), meta.get("page"))
            if key in seen:
                continue
            seen.add(key)

            source_name = str(meta.get("source", "Regulamento"))
            source_clean = re.sub(r"\.(pdf|txt)$", "", source_name,
                                  flags=re.IGNORECASE)
            source_clean = re.sub(r"^(?:data[/\\]raw[/\\]|data[/\\])", "",
                                  source_clean).strip()

            page = meta.get("page")

            # Prefere o item gravado pelo splitter. O regex sobre o texto
            # pegava qualquer sequência numérica — inclusive "31/12/2026"
            # e "R$ 65,00" — e citava item inexistente.
            item = meta.get("item")
            if not item:
                m = re.search(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2}){1,3})\b", doc.page_content)
                item = m.group(1) if m else None

            citation = " — ".join(
                [source_clean]
                + ([f"item {item}"] if item else [])
                + ([f"pag {page}"] if page is not None else [])
            )

            sources.append({
                "title": source_clean,
                "page": page,
                "item": item,
                "url": meta.get("url"),
                "chunk_id": meta.get("chunk_id"),
                "citation": citation,
            })

        return sources[:3]


# ---------------------------------------------------------------- singleton
_service_instance: Optional[AnswerService] = None


def _get_service() -> AnswerService:
    global _service_instance
    if _service_instance is None:
        _service_instance = AnswerService()
    return _service_instance


async def generate_answer(
    question: str,
    documents: List[Document],
    language: str = "pt",
    chat_history: str = "",
) -> Dict[str, Any]:
    """
    Wrapper de compatibilidade. NOTA: a versão anterior não repassava
    chat_history, então qualquer chamador que usasse esta função perdia o
    histórico — e perguntas de acompanhamento ("e na fase presencial?")
    não eram reescritas corretamente. Corrigido.
    """
    return await _get_service().generate_answer(
        question, documents, language, chat_history
    )