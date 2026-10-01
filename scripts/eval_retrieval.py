import json
import os
import sys
import time
import numpy as np

from dotenv import load_dotenv

from rag.db import connect
from rag.eval_data import load_50_questions
from rag.retrieval import embed_query, search

load_dotenv()
K_VALUES = [5, 10, 20]
EXACT = "--exact" in sys.argv
EF = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--ef=")), None)

items = load_50_questions()
rows = []
search_times = []

with connect() as conn:
    if EXACT:
        conn.execute("SET enable_indexscan = off")
    if EF:
        conn.execute(f"SET hnsw.ef_search = {EF}")

    for item in items:
        # results = search(conn, embed_query(item["question"]), k=max(K_VALUES))
        query_vec = embed_query(item["question"])
        t = time.perf_counter()
        results = search(conn, query_vec, k=max(K_VALUES))
        search_times.append(time.perf_counter() - t)
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
out_file = "results/retrieval_exact.json" if EXACT else "results/retrieval_baseline.json"
with open(out_file, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved to {out_file}")

times_ms = np.array(search_times) * 1000
print(f"Search latency: p50 = {np.percentile(times_ms, 50):.0f} ms, p95 = {np.percentile(times_ms, 95):.0f} ms")