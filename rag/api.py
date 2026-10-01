from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from rag.db import connect

load_dotenv()

app = FastAPI(title="CleanTech RAG")


@app.get("/health")
def health():
    """Alive AND able to reach the database. Kubernetes will use this later."""
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {e}")
    return {"status": "ok"}