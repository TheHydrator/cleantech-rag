import numpy as np
from rag.embeddings import embed_texts


def embed_query(question):
    """Turn a question into an embedding, the same way the chunks were embedded."""
    return np.array(embed_texts([question])[0], dtype=np.float32)


def search(conn, query_vec, k=5):
    """Return the k chunks closest in meaning to the query vector."""
    return conn.execute(
        """SELECT c.article_id, c.chunk_index, a.title, c.content,
                  1 - (c.embedding <=> %s) AS similarity
           FROM chunks c
           JOIN articles a ON a.id = c.article_id
           ORDER BY c.embedding <=> %s
           LIMIT %s""",
        (query_vec, query_vec, k),
    ).fetchall()