import os 
from dotenv import load_dotenv
import psycopg
from openai import OpenAI

load_dotenv()

#1. Can we reach Postgres?
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    print("Postgres OK, pgvector version:", version[0])

#2. Does the openai key work?
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
result = client.embeddings.create(model="text-embedding-3-small", input="solar energy")
print("OpenAI OK, embedding length:", len(result.data[0].embedding))