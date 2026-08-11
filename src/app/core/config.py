"""
src/app/core/config.py

CORREÇÃO em relação à versão anterior
-------------------------------------
allowed_origins era List[str]. pydantic-settings decodifica campos de tipo
complexo como JSON NA FONTE, antes de qualquer field_validator rodar — então
o validator com mode="before" nunca era alcançado e
"http://a.com,http://b.com" quebrava o carregamento inteiro.

Agora o campo é str e a lista sai da propriedade .origins, que aceita tanto
CSV quanto JSON. Use settings.origins no CORS do main.py.

DIAGNÓSTICO
    python -m src.app.core.config
Rode nos dois ambientes (local e produção) e compare o fingerprint.
"""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------- credenciais
    openai_api_key: str = Field(...)

    # ---------------------------------------------------------- índice
    faiss_index_path: str = "data/processed/faiss_index"
    embedding_model: str = "text-embedding-3-large"
    distance_strategy: Literal["EUCLIDEAN_DISTANCE", "MAX_INNER_PRODUCT"] = (
        "EUCLIDEAN_DISTANCE"
    )

    # ---------------------------------------------------------- chunking
    # Só afeta scripts/build_index.py. Mudou aqui, precisa reconstruir.
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ---------------------------------------------------------- retrieval
    retrieval_fetch_k: int = 20
    max_retrieve: int = 6
    max_rerank: int = 4

    # Cosseno real: cos = 1 - (1-s)^2 sobre o score do LangChain.
    # NÃO reaproveitar o 0.3 antigo — ele equivalia a cosseno 0.51.
    #   0.744  melhor hit PT medido neste índice
    #   0.592  "como recupero minha senha?"
    #   0.510  <- o que o antigo 0.3 significava
    retrieval_cosine_threshold: float = 0.25
    support_cosine_threshold: float = 0.15

    # Piso: nunca entregar contexto vazio ao gerador.
    min_chunks: int = 2

    # Era 2. Respostas corretas costumam ser chunks contíguos da mesma
    # página, então um teto baixo trunca regras de várias cláusulas.
    max_chunks_per_page: int = 4

    bm25_top_accept: int = 5
    fusion_strategy: Literal["rrf", "weighted"] = "rrf"

    # Corpus é 100% PT. Medido: "how do I recover my password?"
    # cosseno 0.027 em EN -> 0.592 traduzido para PT.
    translate_query_to_pt: bool = True

    # OBSOLETO. Nada deve ler isto após o patch do vectorstore.
    retrieval_score_threshold: float = 0.3
    support_score_threshold: float = 0.12

    # ---------------------------------------------------------- reranker
    # ms-marco-MiniLM-L-6-v2 é treinado só em inglês; reordenava texto em
    # português com quase nenhum sinal.
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_revision: str = "1427fd652930e4ba29e8149678df786c240d8825"

    # ---------------------------------------------------------- geração
    generation_model: str = "gpt-4o-mini"
    generation_temperature: float = 0.1
    # Era 300, teto duplo junto com "2-3 frases": as respostas gold têm
    # 5,08 fatos atômicos e enumerações eram cortadas no meio da frase.
    generation_max_tokens: int = 800

    # ---------------------------------------------------------- calendário
    calendar_url: str = ""
    calendar_yml: str = "data/calendar.yml"
    obg_site: str = "https://obgeografia.com.br"

    # ---------------------------------------------------------- cache
    cache_path: str = "data/processed/.lumie_cache.json"
    cache_ttl_seconds: int = 7 * 24 * 3600
    cache_max_entries: int = 5000
    cache_enabled: bool = True
    history_backend: Literal["memory", "file"] = "memory"

    # ---------------------------------------------------------- app
    # str, NÃO List[str]. pydantic-settings tentaria json.loads() na fonte,
    # antes de qualquer validator, e uma lista separada por vírgula quebra
    # o carregamento inteiro do Settings.
    allowed_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    @property
    def origins(self) -> List[str]:
        """Aceita CSV ou JSON. Use isto no CORSMiddleware."""
        v = (self.allowed_origins or "").strip()
        if not v:
            return []
        if v.startswith("["):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return [o.strip() for o in v.split(",") if o.strip()]

    @field_validator("retrieval_cosine_threshold", "support_cosine_threshold")
    @classmethod
    def _limiar_sensato(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("limiar deve ser um cosseno em [-1, 1]")
        if v > 0.6:
            raise ValueError(
                f"limiar {v} está acima do melhor score observado neste corpus "
                "(0.744) menos folga — isso recusaria quase tudo. Você copiou "
                "o valor 0.3 do espaço antigo por engano?"
            )
        return v

    # ---------------------------------------------------------- diagnóstico
    def index_checksum(self) -> str:
        f = Path(self.faiss_index_path) / "index.faiss"
        try:
            st = f.stat()
            return f"{st.st_size}-{int(st.st_mtime)}"
        except OSError:
            return "AUSENTE"

    def answer_affecting(self) -> dict:
        """Tudo que pode mudar uma resposta. Nada além disso."""
        return {
            "embedding_model": self.embedding_model,
            "distance_strategy": self.distance_strategy,
            "generation_model": self.generation_model,
            "generation_temperature": self.generation_temperature,
            "generation_max_tokens": self.generation_max_tokens,
            "retrieval_fetch_k": self.retrieval_fetch_k,
            "retrieval_cosine_threshold": self.retrieval_cosine_threshold,
            "support_cosine_threshold": self.support_cosine_threshold,
            "min_chunks": self.min_chunks,
            "max_retrieve": self.max_retrieve,
            "max_rerank": self.max_rerank,
            "max_chunks_per_page": self.max_chunks_per_page,
            "bm25_top_accept": self.bm25_top_accept,
            "fusion_strategy": self.fusion_strategy,
            "use_reranker": self.use_reranker,
            "reranker_model": self.reranker_model,
            "reranker_revision": self.reranker_revision,
            "translate_query_to_pt": self.translate_query_to_pt,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_checksum": self.index_checksum(),
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.answer_affecting(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        linhas = [f"config fingerprint: {self.fingerprint()}"]
        for k, v in sorted(self.answer_affecting().items()):
            linhas.append(f"  {k:32} {v}")
        linhas.append(f"  {'origins':32} {self.origins}")
        return "\n".join(linhas)

    def validate_runtime(self) -> None:
        """Chame no startup do main.py. Falhar alto é melhor que silenciar."""
        idx = Path(self.faiss_index_path)
        if not (idx / "index.faiss").exists():
            raise RuntimeError(
                f"Índice FAISS não encontrado em {idx}. "
                f"Rode: python -m scripts.build_index"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("chunk_overlap deve ser menor que chunk_size")
        if self.min_chunks > self.max_retrieve:
            raise RuntimeError("min_chunks não pode exceder max_retrieve")


def _load() -> Settings:
    try:
        return Settings()
    except Exception as e:
        raise SystemExit(
            f"\nErro de configuração: {e}\n\n"
            "Confira o .env. Lembrete: ALLOWED_ORIGINS aceita vírgula\n"
            "  ALLOWED_ORIGINS=http://localhost:3000,https://obgeografia.com.br\n"
        )


settings = _load()


if __name__ == "__main__":
    print(settings.describe())