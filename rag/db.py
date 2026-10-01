import os
import psycopg
from pgvector.psycopg import register_vector


def connect():
    """Open a database connection that understands vectors."""
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    ef = os.environ.get("HNSW_EF_SEARCH")
    if ef:
        conn.execute(f"SET hnsw.ef_search = {int(ef)}")
    return conn