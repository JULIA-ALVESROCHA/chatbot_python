# src/rag_pipeline/rewrite/prompts.py
"""
Prompt templates for query rewriting in RAG systems.

This module contains prompts designed to transform user queries into
self-contained, search-optimized versions that improve retrieval performance.

Based on:
- "Empowering Air Travelers" (NLLP Workshop 2024, Appendix A)
- RAG best practices and query decomposition techniques

Key features:
- Resolves ambiguous pronouns and references
- Incorporates context from conversation history
- Decomposes complex multi-part questions
- Makes queries standalone for better semantic search
- Bilingual support (PT-BR/EN) with automatic detection
"""

from typing import Literal

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

# Número máximo de sub-perguntas geradas na decomposição
MAX_DECOMPOSED_QUERIES = 3

# Temperatura para geração
REWRITE_TEMPERATURE = 0

# Máximo de tokens para respostas de reescrita
MAX_REWRITE_TOKENS = 300


# ==============================================================================
# DETECÇÃO DE IDIOMA
# ==============================================================================

def detect_language(text: str) -> Literal["pt", "en"]:
    """
    Detecta o idioma do texto de forma simples e rápida.
    
    Args:
        text: Texto para detecção de idioma
        
    Returns:
        "pt" para português ou "en" para inglês
        
    Note:
        Usa heurística baseada em palavras comuns. Para casos ambíguos,
        assume português (idioma padrão do sistema).
    """
    text_lower = text.lower()
    
    # Palavras-chave comuns em português
    pt_keywords = [
        "que", "qual", "como", "quando", "onde", "por", "para",
        "regulamento", "prova", "olimpíada", "posso", "pode",
        "devo", "preciso", "está", "são", "foi", "será", "isso",
        "aquilo", "ele", "ela"
    ]
    
    # Palavras-chave comuns em inglês
    en_keywords = [
        "what", "which", "how", "when", "where", "why", "can",
        "regulation", "exam", "olympiad", "should", "must",
        "is", "are", "was", "will", "the", "this", "that", "it"
    ]
    
    pt_count = sum(1 for kw in pt_keywords if kw in text_lower)
    en_count = sum(1 for kw in en_keywords if kw in text_lower)
    
    # Se empate ou nenhuma detecção, assume português (idioma padrão)
    return "pt" if pt_count >= en_count else "en"


# ==============================================================================
# PROMPT 1: DECONTEXTUALIZAÇÃO (Query Rewriting)
# ==============================================================================

QUERY_REWRITE_TEMPLATE_PT = """Você é um assistente de reescrita de perguntas para um sistema RAG (Retrieval-Augmented Generation).

Sua tarefa é reescrever a pergunta do usuário em uma versão autossuficiente e otimizada para busca que melhorará o desempenho de recuperação de documentos sobre regulamentos de olimpíadas brasileiras de Geografia.

Instruções:
1. Se a pergunta contém pronomes (ex: "ele", "ela", "isso", "aquilo") ou referências ao contexto anterior, resolva-os usando o histórico de conversa fornecido abaixo.
2. Torne a pergunta reescrita autossuficiente - ela deve fazer sentido completo mesmo sem o histórico de conversa.
3. Preserve a intenção central e o significado da pergunta original.
4. Expanda termos abreviados ou contexto implícito quando útil para a busca.
5. Se a pergunta já é clara e autossuficiente, você pode retorná-la com pequenas melhorias.
6. NÃO adicione informações que não estão implícitas na pergunta ou no histórico.
7. Mantenha a pergunta reescrita concisa e focada.
8. Responda APENAS com a pergunta reescrita, sem explicações adicionais.
9. If the information is not present in the OBG official documents, explicitly state that it is not mentioned in the edital or regulamento.
Histórico da Conversa:
{chat_history}

Pergunta Original:
{question}

Pergunta Reescrita (autossuficiente, otimizada para busca semântica):"""

QUERY_REWRITE_TEMPLATE_EN = """You are a query rewriting assistant for a RAG (Retrieval-Augmented Generation) system.

Your task is to rewrite the user's question into a self-contained, search-optimized version that will improve document retrieval performance for Brazilian Geography Olympiad regulations.

Instructions:
1. If the question contains pronouns (e.g., "it", "this", "that", "they") or references to previous context, resolve them using the chat history provided below.
2. Make the rewritten question standalone - it should make complete sense even without the chat history.
3. Preserve the core intent and meaning of the original question.
4. Expand abbreviated terms or implicit context when helpful for search.
5. If the question is already clear and self-contained, you may return it with minor improvements.
6. Do NOT add information that isn't implied by the question or chat history.
7. Keep the rewritten question concise and focused.
8. Answer ONLY with the rewritten question, without additional explanations.
9. If the information is not present in the OBG official documents, explicitly state that it is not mentioned in the edital or regulamento.
Chat History:
{chat_history}

Original Question:
{question}

Rewritten Question (self-contained, optimized for semantic search):"""


# ==============================================================================
# PROMPT 1B: REESCRITA MÍNIMA (sem histórico disponível)
# ==============================================================================

QUERY_REWRITE_MINIMAL_TEMPLATE_PT = """Reescreva a seguinte pergunta para ser autossuficiente e otimizada para busca semântica de documentos sobre regulamentos de olimpíadas brasileiras de Geografia.

Remova pronomes ambíguos, expanda contexto implícito e garanta que a pergunta seja clara sem informações adicionais.

Responda APENAS com a pergunta reescrita, sem explicações adicionais.

Pergunta Original:
{question}

Pergunta Reescrita:"""

QUERY_REWRITE_MINIMAL_TEMPLATE_EN = """Rewrite the following question to be self-contained and optimized for semantic document search about Brazilian Geography Olympiad regulations.

Remove ambiguous pronouns, expand implicit context, and ensure the question is clear without additional information.

Answer ONLY with the rewritten question, without additional explanations.

Original Question:
{question}

Rewritten Question:"""


# ==============================================================================
# PROMPT 2: DECOMPOSIÇÃO (Decompositional Query Generation)
# ==============================================================================

DECOMPOSITION_TEMPLATE_PT = """Você é um assistente especializado em decompor perguntas complexas sobre regulamentos de olimpíadas brasileiras de Geografia.

Sua tarefa é identificar as informações necessárias para responder à pergunta do usuário e dividi-la em sub-perguntas mais simples e específicas.

Instruções:
1. Forneça sua resposta como uma lista numerada de perguntas.
2. Cada pergunta deve focar em um único aspecto respondível da entrada.
3. Limite a lista a no máximo {max_queries} perguntas.
4. Se a pergunta já for simples e direta, retorne apenas ela mesma como uma única pergunta.
5. As sub-perguntas devem ser independentes e claras.
6. Mantenha o mesmo idioma da pergunta original.
7. NÃO adicione numeração extra ou texto explicativo, apenas as perguntas numeradas.

Pergunta:
{query}

Sub-perguntas (uma por linha, numeradas 1., 2., 3.):"""

DECOMPOSITION_TEMPLATE_EN = """You are an assistant specialized in decomposing complex questions about Brazilian Geography Olympiad regulations.

Your task is to identify the information needed to respond to the user's question and break it down into simpler, more specific sub-questions.

Instructions:
1. Provide your answer as a numbered list of questions.
2. Each question should focus on a single, answerable aspect of the input.
3. Limit the list to a maximum of {max_queries} questions.
4. If the question is already simple and direct, return only itself as a single question.
5. The sub-questions should be independent and clear.
6. Maintain the same language as the original question.
7. Do NOT add extra numbering or explanatory text, just the numbered questions.

Question:
{query}

Sub-questions (one per line, numbered 1., 2., 3.):"""


# ==============================================================================
# FUNÇÕES DE FORMATAÇÃO DE PROMPTS
# ==============================================================================

def get_query_rewrite_prompt(
    question: str,
    chat_history: str = "",
    language: str = "pt",
    use_minimal: bool = False
) -> str:
    """
    Formata o prompt de reescrita de query (decontextualização).
    
    Args:
        question: Pergunta original do usuário
        chat_history: Histórico de conversa formatado como string
        language: Idioma ("pt" ou "en")
        use_minimal: Se True, usa template minimal (sem histórico)
        
    Returns:
        Prompt formatado pronto para envio ao LLM
        
    Example:
        >>> # Com histórico
        >>> prompt = get_query_rewrite_prompt(
        ...     question="E sobre isso?",
        ...     chat_history="User: Qual é a duração da prova?\nAssistant: A prova tem 3 horas.",
        ...     language="pt"
        ... )
        >>> # Sem histórico (minimal)
        >>> prompt = get_query_rewrite_prompt(
        ...     question="Quais são os critérios?",
        ...     use_minimal=True
        ... )
    """
    # Se não há histórico ou use_minimal=True, usa template minimal
    if use_minimal or not chat_history or chat_history.strip() == "":
        template = (
            QUERY_REWRITE_MINIMAL_TEMPLATE_PT if language == "pt"
            else QUERY_REWRITE_MINIMAL_TEMPLATE_EN
        )
        return template.format(question=question)
    
    # Usa template completo com histórico
    template = (
        QUERY_REWRITE_TEMPLATE_PT if language == "pt"
        else QUERY_REWRITE_TEMPLATE_EN
    )
    
    return template.format(
        question=question,
        chat_history=chat_history
    )


def get_decomposition_prompt(
    query: str,
    language: str = "pt",
    max_queries: int = MAX_DECOMPOSED_QUERIES
) -> str:
    """
    Formata o prompt de decomposição para dividir perguntas complexas.
    
    Args:
        query: Pergunta (já decontextualizada) para decompor
        language: Idioma ("pt" ou "en")
        max_queries: Número máximo de sub-perguntas (padrão: 3)
        
    Returns:
        Prompt formatado pronto para envio ao LLM
        
    Example:
        >>> prompt = get_decomposition_prompt(
        ...     query="Quais são os critérios de inscrição e como funciona a prova?",
        ...     language="pt"
        ... )
    """
    template = (
        DECOMPOSITION_TEMPLATE_PT if language == "pt"
        else DECOMPOSITION_TEMPLATE_EN
    )
    
    return template.format(
        query=query,
        max_queries=max_queries
    )


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================

def format_chat_history(history: list[dict]) -> str:
    """
    Formata histórico de conversa para inclusão no prompt.
    
    Args:
        history: Lista de dicts com formato [{"role": "user", "content": "..."}, ...]
        
    Returns:
        String formatada do histórico
        
    Example:
        >>> history = [
        ...     {"role": "user", "content": "Qual a duração da prova?"},
        ...     {"role": "assistant", "content": "A prova tem 3 horas."}
        ... ]
        >>> print(format_chat_history(history))
        User: Qual a duração da prova?
        Assistant: A prova tem 3 horas.
    """
    if not history:
        return ""
    
    formatted_lines = []
    for turn in history:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        formatted_lines.append(f"{role}: {content}")
    
    return "\n".join(formatted_lines)


def parse_decomposed_queries(llm_response: str) -> list[str]:
    """
    Extrai lista de perguntas da resposta do LLM no prompt de decomposição.
    
    Args:
        llm_response: Resposta bruta do modelo (texto com perguntas numeradas)
        
    Returns:
        Lista de perguntas limpas (sem numeração)
        
    Example:
        >>> response = "1. Quais são os critérios?\n2. Como é a prova?"
        >>> parse_decomposed_queries(response)
        ['Quais são os critérios?', 'Como é a prova?']
    """
    lines = llm_response.strip().split("\n")
    queries = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numeração (1., 2., 1), 2), etc.)
        import re
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        
        if cleaned:
            queries.append(cleaned)
    
    return queries[:MAX_DECOMPOSED_QUERIES]  # Garante limite máximo


def should_use_minimal_prompt(chat_history: str) -> bool:
    """
    Determina se deve usar prompt minimal (economia de tokens).
    
    Args:
        chat_history: Histórico de conversa
        
    Returns:
        True se deve usar minimal, False caso contrário
        
    Note:
        Usa prompt minimal quando não há histórico ou é muito curto,
        economizando tokens sem perda de qualidade.
    """
    if not chat_history or chat_history.strip() == "":
        return True
    
    # Se histórico tem menos de 50 caracteres, provavelmente é inútil
    if len(chat_history.strip()) < 50:
        return True
    
    return False


# ==============================================================================
# MENSAGENS DE FALLBACK
# ==============================================================================

FALLBACK_MESSAGES = {
    "pt": {
        "empty_query": "Por favor, faça uma pergunta sobre os regulamentos da Olimpíada Brasileira de Geografia.",
        "invalid_query": "Desculpe, não consegui entender sua pergunta. Pode reformular?",
        "no_decomposition": "Não foi possível decompor a pergunta. Processando como pergunta simples.",
        "rewrite_failed": "Erro ao processar a pergunta. Usando versão original.",
    },
    "en": {
        "empty_query": "Please ask a question about Brazilian Geography Olympiad regulations.",
        "invalid_query": "Sorry, I couldn't understand your question. Could you rephrase it?",
        "no_decomposition": "Could not decompose the question. Processing as simple question.",
        "rewrite_failed": "Error processing question. Using original version.",
    }
}


def get_fallback_message(message_type: str, language: str = "pt") -> str:
    """
    Retorna mensagem de fallback apropriada.
    
    Args:
        message_type: Tipo da mensagem ("empty_query", "invalid_query", etc.)
        language: Idioma ("pt" ou "en")
        
    Returns:
        Mensagem de fallback formatada
    """
    return FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES["pt"]).get(
        message_type,
        FALLBACK_MESSAGES["pt"]["invalid_query"]
    )


# ==============================================================================
# VALIDAÇÃO DE QUERIES (Opcional)
# ==============================================================================

def is_valid_query(query: str, min_length: int = 3) -> bool:
    """
    Valida se a query é minimamente aceitável.
    
    Args:
        query: Query para validar
        min_length: Comprimento mínimo em caracteres
        
    Returns:
        True se válida, False caso contrário
        
    Example:
        >>> is_valid_query("Quais são os critérios?")
        True
        >>> is_valid_query("?")
        False
    """
    if not query or not isinstance(query, str):
        return False
    
    query_clean = query.strip()
    
    # Muito curta
    if len(query_clean) < min_length:
        return False
    
    # Apenas pontuação
    if all(c in "?!.,;: " for c in query_clean):
        return False
    
    return True