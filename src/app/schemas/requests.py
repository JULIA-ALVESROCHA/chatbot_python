from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    """
    Representa uma mensagem individual do histórico da conversa.
    Usado para dar contexto ao pipeline (rewrite / generation).
    """
    role: str = Field(
        ...,
        description="Papel da mensagem na conversa",
        examples=["user", "assistant"]
    )
    content: str = Field(
        ...,
        description="Conteúdo textual da mensagem",
        min_length=1,
        max_length=2000
    )


class ChatRequest(BaseModel):
    """
    Request principal do endpoint /api/v1/chat
    """

    question: str = Field(
        ...,
        description="Pergunta atual do usuário",
        min_length=1,
        max_length=512  # safe default
    )

    session_id: Optional[str] = Field(
        None,
        description="Identificador da sessão para manter contexto entre mensagens"
    )

    chat_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Histórico da conversa (mensagens anteriores)"
    )

    # metadata opcional para API pública
    client_id: Optional[str] = Field(
        None,
        description="Identificador do cliente/aplicação consumidora da API"
    )

    request_id: Optional[str] = Field(
        None,
        description="ID único da requisição (debug, tracing, logs)"
    )

    @validator("chat_history", pre=True)
    def limit_chat_history(cls, value):
        """
        Limita o tamanho do histórico para evitar contextos gigantes
        """
        if value and len(value) > 20:
            raise ValueError("chat_history não pode ter mais de 20 mensagens")
        return value
