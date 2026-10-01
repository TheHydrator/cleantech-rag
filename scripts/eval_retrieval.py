import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

from rag.db import connect
from rag.eval_data import get_question_vectors, load_50_questions
from rag.retrieval import search

load_dotenv()
K_VALUES = [5, 10, 20]
EXACT = "--exact" in sys.argv
RERANK = "--rerank" in sys.argv
EF = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--ef=")), None)

if RERANK:
    from rag.rerank import rerank

items = load_50_questions()
vectors = get_question_vectors(items)
rows, search_times, rerank_times = [], [], []

with connect() as conn:
    if EXACT:
        conn.execute("SET enable_indexscan = off")
    if EF:
        conn.execute(f"SET hnsw.ef_search = {EF}")

    for item, query_vec in zip(items, vectors):
        t = time.perf_counter()
        results = search(conn, query_vec, k=max(K_VALUES))
        search_times.append(time.perf_counter() - t)

        if RERANK:
            t = time.perf_counter()
            results = rerank(item["question"], results, top_n=len(results))  # reorder all 20
            rerank_times.append(time.perf_counter() - t)

        ranked_articles = [r[0] for r in results]
        expected = set(item["article_ids"])

        row = {"num": item["num"], "expected": sorted(expected)}
        for k in K_VALUES:
            found = expected & set(ranked_articles[:k])
            row[f"recall@{k}"] = len(found) / len(expected)
        rows.append(row)
        print(f"Q{item['num']:>2}: " + "  ".join(f"@{k}={row[f'recall@{k}']:.2f}" for k in K_VALUES))

print("\n=== Average recall over", len(rows), "questions ===")
for k in K_VALUES:
    avg = sum(r[f"recall@{k}"] for r in rows) / len(rows)
    print(f"recall@{k}: {avg:.3f}")

os.makedirs("results", exist_ok=True)
if EXACT:
    out_file = "results/retrieval_exact.json"
elif RERANK:
    out_file = "results/retrieval_rerank.json"
else:
    out_file = "results/retrieval_baseline.json"
with open(out_file, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved to {out_file}")

search_ms = np.array(search_times) * 1000
print(f"Search latency: p50 = {np.percentile(search_ms, 50):.0f} ms, p95 = {np.percentile(search_ms, 95):.0f} ms")
if RERANK:
    rerank_ms = np.array(rerank_times) * 1000
    print(f"Rerank latency: p50 = {np.percentile(rerank_ms, 50):.0f} ms, p95 = {np.percentile(rerank_ms, 95):.0f} ms")