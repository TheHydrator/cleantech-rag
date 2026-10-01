from dotenv import load_dotenv

from rag.db import connect
from rag.eval_data import load_50_questions

load_dotenv()
items = load_50_questions()
print(f"Parsed {len(items)} questions")

all_ids = {i for item in items for i in item["article_ids"]}
with connect() as conn:
    found = {row[0] for row in conn.execute("SELECT id FROM articles WHERE id = ANY(%s)", (list(all_ids),))}

missing = all_ids - found
print(f"Referenced articles: {len(all_ids)}, found in database: {len(found)}, missing: {sorted(missing)}")
print("Example:", items[0]["num"], items[0]["question"][:80], items[0]["article_ids"])