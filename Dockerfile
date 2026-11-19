FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APEX_PORT=7860 \
    APEX_HOST=0.0.0.0 \
    APEX_EMBED_MODEL=/app/models/bge-large-en-v1.5 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# System build tools for native wheels (hnswlib/chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt ./

# Install CPU-only PyTorch, spacy, and all project dependencies in one pass
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch spacy -r requirements.txt \
 && pip uninstall -y torchvision || true

# Pre-install spaCy NER model for air-gapped PII scrubbing
RUN python -m spacy download en_core_web_lg

# Copy project source (including ./models/bge-large-en-v1.5)
COPY . /app

# Smoke-test offline embedding load during build so runtime never fails
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('/app/models/bge-large-en-v1.5')"

VOLUME ["/app/apex_data"]
EXPOSE 7860

CMD ["python", "app.py"]
