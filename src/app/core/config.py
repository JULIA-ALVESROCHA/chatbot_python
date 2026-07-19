from pydantic_settings import BaseSettings #transform the class in a variable loader
from pydantic import Field
from typing import List



class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY") #.. means required
    faiss_index_path: str = Field("data/processed/faiss_index", env="FAISS_INDEX_PATH")
    max_retrieve: int = Field(6, env="MAX_RETRIEVE")
    max_rerank: int = Field(4, env="MAX_RERANK")
    use_reranker: bool = Field(False, env="USE_RERANKER")
    ALLOWED_ORIGINS: List[str] = Field(default_factory=list)

    # chunking parameters
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")
    chunk_size: int = Field(500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    retrieval_score_threshold: float = Field(0.3, env="RETRIEVAL_SCORE_THRESHOLD")
    retrieval_fetch_k: int = Field(20, env="RETRIEVAL_FETCH_K")
    support_score_threshold: float = Field(0.12, env="SUPPORT_SCORE_THRESHOLD")
    bm25_top_accept: int = Field(5, env="BM25_TOP_ACCEPT")
    max_chunks_per_page: int = Field(2, env="MAX_CHUNKS_PER_PAGE")

    # generation parameters
    generation_model: str = Field("gpt-4o-mini", env="GENERATION_MODEL")
    generation_temperature: float = Field(0.1, env="GENERATION_TEMPERATURE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() #create a global object