import os
import psycopg
from pgvector.psycopg import register_vector


def connect():
    """Open a database connection that understands vectors."""
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn