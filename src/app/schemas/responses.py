from typing import List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Representa uma fonte usada pelo RAG.
    Normalmente vem do metadata do documento recuperado.
    """
    source_id: Optional[str] = Field(
        None,
        description="Identificador da fonte (ex: nome do arquivo, id interno)"
    )

    page: Optional[int] = Field(
        None,
        description="Página ou seção da fonte original"
    )

    excerpt: Optional[str] = Field(
        None,
        description="Trecho curto do texto usado como evidência"
    )


class ChatResponse(BaseModel):
    """
    Response padrão do endpoint /api/v1/chat
    """

    answer: str = Field(
        ...,
        description="Resposta final gerada pelo chatbot"
    )

    sources: List[Source] = Field(
        default_factory=list,
        description="Lista de fontes utilizadas para gerar a resposta"
    )

    session_id: Optional[str] = Field(
        None,
        description="Session id retornado para o cliente"
    )

    language: str = Field(
        default="pt",
        description="Idioma detectado automaticamente da pergunta"
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confiança estimada da resposta (opcional)"
    )

    warnings: Optional[List[str]] = Field(
        default_factory=list,
        description="Avisos ou observações (ex: resposta incompleta, pouca evidência)"
    )
