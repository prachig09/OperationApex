FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APEX_PORT=7860 \
    APEX_HOST=0.0.0.0 \
    APEX_EMBED_MODEL=/app/models/bge-large-en-v1.5 \
    APEX_LLM_URL=http://127.0.0.1:11434 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Install build tools + curl/zstd for Ollama installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates zstd && rm -rf /var/lib/apt/lists/*

# Install Ollama inside the image for self-contained local inference
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install CPU-only PyTorch, spacy, and project dependencies
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch spacy huggingface_hub -r requirements.txt \
 && pip uninstall -y torchvision || true

# Pre-install spaCy NER model for air-gapped PII scrubbing
RUN python -m spacy download en_core_web_lg

# Copy project source
COPY . /app

# Download BGE-large weights at build time if omitted from Git (>100MB GitHub limit)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-large-en-v1.5', local_dir='/app/models/bge-large-en-v1.5')"

# Pre-pull quantized Ollama model into image layer (change model tag if your config uses another)
RUN ollama serve & sleep 5 && ollama pull qwen2.5:3b && pkill ollama

# Enforce strict offline mode at runtime after build downloads complete
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860

# Start local Ollama daemon on 127.0.0.1:11434, wait for readiness, then launch AegisDeal on Render's $PORT
CMD ["sh", "-c", "ollama serve & sleep 3 && APEX_PORT=${PORT:-7860} GRADIO_SERVER_PORT=${PORT:-7860} python app.py"]
