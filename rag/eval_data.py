import json
import os
import re

import numpy as np

from rag.embeddings import embed_texts

CACHE_PATH = "eval_cache/question_embeddings.json"


def load_50_questions(path="data/CleanTech-50answers.txt"):
    """Parse the 50-question file into a list of dicts."""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()

    pattern = re.compile(
        r"\*\*(\d+)\. Query:\*\*\s*(.+?)\s*\n"
        r"\*\*Desired Output:\*\*\s*(.+?)\s*\n"
        r"\*\*Referenced Articles:\*\*\s*([\d,\s]+)",
        re.S,
    )
    return [
        {
            "num": int(num),
            "question": question.strip(),
            "expected_answer": answer.strip(),
            "article_ids": [int(x) for x in re.findall(r"\d+", ids)],
        }
        for num, question, answer, ids in pattern.findall(text)
    ]


def get_question_vectors(items):
    """Return one embedding per question, embedding only the ones not cached yet."""
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)

    missing = [item["question"] for item in items if item["question"] not in cache]
    if missing:
        cache.update(zip(missing, embed_texts(missing)))
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
        print(f"Embedded and cached {len(missing)} new questions")

    return [np.array(cache[item["question"]], dtype=np.float32) for item in items]