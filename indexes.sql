-- Build after loading data: building once at the end is much faster
-- than updating the index on every insert.
SET maintenance_work_mem = '1GB';

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);