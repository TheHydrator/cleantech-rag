import pandas as pd
from rag.chunking import chunk_article

df = pd.read_csv("data/cleantech_media_dataset_v3_2024-10-28.csv")

total = 0
skipped = 0
for content in df["content"]:
    chunks = chunk_article(content)
    if not chunks:
        skipped += 1
    total += len(chunks)

print(f"Articles skipped (too short): {skipped}")
print(f"Total chunks: {total}  (notebook had 131587)")
print("First chunk preview:", chunk_article(df["content"].iloc[0])[0][:200])