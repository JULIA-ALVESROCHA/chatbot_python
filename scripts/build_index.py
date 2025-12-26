from pathlib import Path

import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Pastas
ITEM_REGEX = re.compile(r"\b(\d+\.\d+\.\d+)\b")
DATA_RAW = Path("data/raw")
PROCESSED = Path("data/processed/faiss_index")
PROCESSED.mkdir(parents=True, exist_ok=True)

def load_documents():
    docs = []
    for p in DATA_RAW.iterdir():
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf8")
            docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=800, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(docs)

    # 🔹 NOVO: extrair item do regulamento (ex: 4.2.1, 5.7.3)
    item_pattern = re.compile(r"\b\d+\.\d+\.\d+\b")

    for doc in split_docs:
        text = doc.page_content or ""

        match = item_pattern.search(text)
        if match:
            # salva o item no metadata
            doc.metadata["item"] = match.group()

    return split_docs

def build_faiss(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(PROCESSED))
    return vectorstore

def verify_search(vectorstore):
    results = vectorstore.similarity_search("Quem pode participar?", k=3)
    print("\nRESULTADOS DE TESTE:")
    for r in results:
        print("----")
        print(r.page_content[:300], "...")

if __name__ == "__main__":
    print("1) Carregando documentos...")
    docs = load_documents()
    print(f"Documentos carregados: {len(docs)}")

    print("2) Dividindo documentos...")
    chunks = split_documents(docs)
    print(f"Chunks criados: {len(chunks)}")

    print("3) Criando índice FAISS...")
    vs = build_faiss(chunks)

    print("4) Testando busca...")
    verify_search(vs)

    print("\n✔ INDEXAÇÃO FINALIZADA COM SUCESSO!")
