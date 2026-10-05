import logging

from psycopg.types.json import Jsonb

from rag.db import connect

logger = logging.getLogger("rag.query_log")


def log_query(question, latency_ms, status, answer=None, grade=None, error=None,
              tokens_in=None, tokens_out=None, article_ids=None, settings=None):
    """Write one row to query_logs. Never raises: logging must not break a user request."""
    try:
        with connect() as conn:
            conn.execute(
                """INSERT INTO query_logs
                   (question, answer, grade, status, error, latency_ms,
                    tokens_in, tokens_out, retrieved_article_ids, settings)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (question, answer, grade, status, error, latency_ms,
                 tokens_in, tokens_out, article_ids,
                 Jsonb(settings) if settings else None),
            )
    except Exception:
        logger.exception("failed to write query log")