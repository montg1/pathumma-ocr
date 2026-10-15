FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY server.py .
COPY web ./web

RUN useradd --create-home --uid 1000 app
USER app
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
# /health is 503 until vLLM has loaded the model
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"

CMD ["python", "server.py"]
