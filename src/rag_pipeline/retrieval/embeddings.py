from langchain_openai import OpenAIEmbeddings


def get_embeddings(
    model: str = "text-embedding-3-large",
):
    """
    Create and return an OpenAI embeddings model.

    Args:
        model (str): OpenAI embedding model name.

    Returns:
        OpenAIEmbeddings: Embeddings instance.
    """
    embeddings = OpenAIEmbeddings(
        model=model,
    )

    print(f"[EMBEDDINGS] Using OpenAI embeddings model: {model}")

    return embeddings
