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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() #create a global object

