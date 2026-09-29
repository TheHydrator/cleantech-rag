import os
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

_client = None


def get_client():
    global _client
    if _client is None:
        # max_retries: if OpenAI says "slow down", wait and try again, up to 5 times
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5)
    return _client


def embed_texts(texts):
    """Embed a list of texts. Raises an error instead of ever returning fake vectors."""
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    if len(vectors) != len(texts):
        raise RuntimeError(f"Sent {len(texts)} texts but got {len(vectors)} embeddings back")
    for v in vectors:
        if len(v) != EMBEDDING_DIM:
            raise RuntimeError(f"Expected {EMBEDDING_DIM} dims, got {len(v)}")
    return vectors