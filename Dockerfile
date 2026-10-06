FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only PyTorch: the default version bundles GPU libraries and is several GB larger
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the reranker at build time, so the container never needs the internet to start
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# The model is already in the image: never contact Hugging Face at runtime
ENV HF_HUB_OFFLINE=1

COPY rag ./rag

EXPOSE 8000
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]