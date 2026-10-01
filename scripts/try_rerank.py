from dotenv import load_dotenv

from rag.db import connect
from rag.rerank import rerank
from rag.retrieval import embed_query, search

load_dotenv()
question = "Where do the turbines used in Icelandic geothermal power plants come from?"

with connect() as conn:
    top20 = search(conn, embed_query(question), k=20)

print("BEFORE (embedding order, top 20):")
for i, r in enumerate(top20, 1):
    print(f"  {i:>2}. article {r[0]}  {r[2][:60]}")

print("\nAFTER (reranked, top 5):")
for i, r in enumerate(rerank(question, top20, top_n=5), 1):
    print(f"  {i}. article {r[0]}  {r[2][:60]}")