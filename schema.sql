CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    published   DATE,
    domain      TEXT,
    url         TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    article_id   INTEGER NOT NULL REFERENCES articles(id),
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(1536) NOT NULL,
    UNIQUE (article_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS query_logs (
    id                     BIGSERIAL PRIMARY KEY,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    question               TEXT NOT NULL,
    answer                 TEXT,
    grade                  TEXT,
    status                 TEXT NOT NULL,      -- 'ok' or 'error'
    error                  TEXT,
    latency_ms             INTEGER NOT NULL,
    tokens_in              INTEGER,
    tokens_out             INTEGER,
    retrieved_article_ids  INTEGER[],
    settings               JSONB
);

CREATE INDEX IF NOT EXISTS query_logs_created_at_idx ON query_logs (created_at);