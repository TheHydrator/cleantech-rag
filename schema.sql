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