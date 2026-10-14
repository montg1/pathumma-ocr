FROM python:3.12-slim

# gcc: Triton (used by flash-linear-attention) JIT-compiles a small C launcher at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

WORKDIR /app

# torch ships its own CUDA runtime; only the host NVIDIA driver is needed
RUN uv pip install --system --no-cache \
        torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY server.py .
COPY web ./web

# model weights (HF cache) and compiled Triton kernels live in a volume at /data
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data
USER app
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    TRITON_CACHE_DIR=/data/triton

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["python", "server.py"]
