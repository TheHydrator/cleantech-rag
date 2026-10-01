import os
from sentence_transformers import CrossEncoder

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)  # downloads the model on first use
    return _model


def rerank(question, results, top_n=5):
    """Re-score each (question, chunk) pair and keep the best top_n."""
    scores = get_model().predict([(question, r[3]) for r in results])
    ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)
    return [r for r, _ in ranked[:top_n]]