from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

class StorageManager:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def vectorstore_path(self, name: str) -> Path:
        return self.base_path / name

    def vectorstore_exists(self, name: str) -> bool:
        return self.vectorstore_path(name).exists()

    def save_vectorstore(self, name: str, vectorstore: FAISS):
        path = self.vectorstore_path(name)
        vectorstore.save_local(str(path))

    def load_vectorstore(
        self,
        name: str,
        embeddings: Embeddings
    ) -> Optional[FAISS]:
        path = self.vectorstore_path(name)
        if not path.exists():
            return None

        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True
        )
