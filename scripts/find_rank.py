import sys
from dotenv import load_dotenv

from rag.db import connect
from rag.retrieval import embed_query

load_dotenv()
article_id = int(sys.argv[1])
question = " ".join(sys.argv[2:])

vec = embed_query(question)
with connect() as conn:
    rows = conn.execute(
        """WITH ranked AS (
               SELECT article_id, chunk_index, content,
                      1 - (embedding <=> %s) AS similarity,
                      RANK() OVER (ORDER BY embedding <=> %s) AS rank
               FROM chunks
           )
           SELECT chunk_index, similarity, rank, content
           FROM ranked WHERE article_id = %s ORDER BY rank""",
        (vec, vec, article_id),
    ).fetchall()

print(f"Question: {question}\nArticle {article_id} has {len(rows)} chunks:\n")
for chunk_index, similarity, rank, content in rows:
    print(f"chunk {chunk_index}: rank {rank} of 131587, similarity {similarity:.3f}")
    print(f"    {content[:300]}...\n")