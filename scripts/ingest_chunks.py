import argparse
import os
import time

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from rag.chunking import chunk_article
from rag.embeddings import embed_texts

load_dotenv()
CSV_PATH = "data/cleantech_media_dataset_v3_2024-10-28.csv"
BATCH_SIZE = 200  # roughly how many chunks per OpenAI call

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="Only process this many articles (for testing)")
args = parser.parse_args()

df = pd.read_csv(CSV_PATH).rename(columns={"Unnamed: 0": "id"})

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    register_vector(conn)

    # Resume support: skip articles that already have chunks saved
    done = {row[0] for row in conn.execute("SELECT DISTINCT article_id FROM chunks")}
    todo = df[~df["id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Already done: {len(done)} articles. Processing now: {len(todo)}")

    def save_batch(batch):
        vectors = embed_texts([text for _, _, text in batch])
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (article_id, chunk_index, content, embedding)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (article_id, chunk_index) DO NOTHING""",
                [(a, i, t, np.array(v, dtype=np.float32)) for (a, i, t), v in zip(batch, vectors)],
            )
        conn.commit()

    batch, saved, batches, start = [], 0, 0, time.time()
    for article_id, content in zip(todo["id"], todo["content"]):
        # All chunks of one article go into the same batch, so an article is
        # either fully saved or not saved at all
        batch.extend((int(article_id), i, text) for i, text in enumerate(chunk_article(content)))
        if len(batch) >= BATCH_SIZE:
            save_batch(batch)
            saved, batches, batch = saved + len(batch), batches + 1, []
            if batches % 10 == 0:
                print(f"Saved {saved} chunks ({time.time() - start:.0f}s)")

    if batch:
        save_batch(batch)
        saved += len(batch)

    print(f"Done. Saved {saved} chunks this run in {time.time() - start:.0f}s")