import os
import sys
import time

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.retrieval import embed_query, search

load_dotenv()

args = sys.argv[1:]
no_index = "--no-index" in args
args = [a for a in args if a != "--no-index"]
question = " ".join(args) or "What is the EU's Green Deal Industrial Plan?"

t0 = time.time()
query_vec = embed_query(question)
t1 = time.time()

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    if no_index:
        conn.execute("SET enable_indexscan = off")  # force the old one-by-one scan
    t2 = time.time()
    results = search(conn, query_vec)
    t3 = time.time()

print(f"Question: {question}\n")

for article_id, chunk_index, title, content, similarity, *_ in results:
    print(f"[{similarity:.3f}] article {article_id}, chunk {chunk_index}: {title}")
    print(f"    {content[:150]}...\n")

print(f"Embedding the question: {t1 - t0:.2f}s")
print(f"Connecting to Postgres: {t2 - t1:.3f}s")
print(f"Search only:            {t3 - t2:.3f}s  ({'NO index' if no_index else 'with index'})")