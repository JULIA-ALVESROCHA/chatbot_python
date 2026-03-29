from langchain_openai import OpenAIEmbeddings


def get_embeddings(
    model: str = "text-embedding-3-large",
):
    embeddings = OpenAIEmbeddings(
        model=model,
    )

    print(f"[EMBEDDINGS] Using OpenAI embeddings model: {model}")

    return embeddings
