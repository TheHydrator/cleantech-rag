import os
import sys
import time

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.generation import generate_answer
from rag.retrieval import embed_query, search

load_dotenv()
question = " ".join(sys.argv[1:]) or "What is the EU's Green Deal Industrial Plan?"

t0 = time.time()
query_vec = embed_query(question)
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)
    results = search(conn, query_vec)
t1 = time.time()
answer, usage = generate_answer(question, results)
t2 = time.time()

print(f"Question: {question}\n")
print(f"Answer:\n{answer}\n")
print("Sources:", ", ".join(f"{r[0]} ({r[4]:.2f})" for r in results))
print(f"\nRetrieval: {t1 - t0:.2f}s | Generation: {t2 - t1:.2f}s")
print(f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")