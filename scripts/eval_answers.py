import argparse
import json
import os
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from rag.eval_data import load_50_questions
from rag.graph import build_graph
from rag.judge import judge_answer

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="Name for this run, used in the results file")
parser.add_argument("--top-k", type=int, default=5)
parser.add_argument("--rerank", action="store_true")
parser.add_argument("--fetch-k", type=int, default=20)
parser.add_argument("--hyde", action="store_true")
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

items = load_50_questions()[: args.limit]
app = build_graph()
config = {"configurable": {
    "use_hyde": args.hyde, "top_k": args.top_k, "rerank": args.rerank, "fetch_k": args.fetch_k,
}}

rows = []
for item in items:
    t = time.perf_counter()
    state = app.invoke({"query": item["question"]}, config=config)
    latency = time.perf_counter() - t
    answer = state["final_answer"]

    try:
        score, reason, judge_usage = judge_answer(item["question"], item["expected_answer"], answer)
    except Exception as e:
        score, reason, judge_usage = None, f"JUDGE ERROR: {e}", None

    expected = set(item["article_ids"])
    found = expected & {r[0] for r in state["retrieved"]}
    row = {
        "num": item["num"],
        "score": score,
        "reason": reason,
        "refused": "i don't know" in answer.lower(),
        "recall": len(found) / len(expected),
        "latency_s": round(latency, 2),
        "tokens_in": state["usage"]["in"],
        "tokens_out": state["usage"]["out"],
        "answer": answer,
    }
    rows.append(row)
    print(f"Q{row['num']:>2}: score={score}  recall={row['recall']:.2f}  "
          f"{row['latency_s']}s  tokens_in={row['tokens_in']}")

scored = [r["score"] for r in rows if r["score"] is not None]
lat = np.array([r["latency_s"] for r in rows])
summary = {
    "questions": len(rows),
    "avg_score": round(float(np.mean(scored)), 2) if scored else None,
    "judge_errors": len(rows) - len(scored),
    "refusals": sum(r["refused"] for r in rows),
    "avg_recall": round(float(np.mean([r["recall"] for r in rows])), 3),
    "latency_p50_s": round(float(np.percentile(lat, 50)), 1),
    "latency_p95_s": round(float(np.percentile(lat, 95)), 1),
    "avg_tokens_in": round(float(np.mean([r["tokens_in"] for r in rows]))),
    "avg_tokens_out": round(float(np.mean([r["tokens_out"] for r in rows]))),
}

print("\n=== Summary ===")
for key, value in summary.items():
    print(f"{key}: {value}")

os.makedirs("results", exist_ok=True)
out_file = f"results/answers_{args.name}.json"
with open(out_file, "w") as f:
    json.dump({"config": vars(args), "summary": summary, "rows": rows}, f, indent=2)
print(f"\nSaved to {out_file}")