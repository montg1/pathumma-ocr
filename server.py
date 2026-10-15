"""OCR front-end for nectec/Pathumma-llm-vision-3.0.0-preview served by vLLM.

Inference runs in a vLLM server (VLLM_URL); this app adds the web UI, image downscaling,
PDF splitting and the /ocr upload endpoint, and forwards chat requests to vLLM.

Endpoints
  GET  /                       - web UI: upload images and OCR them
  GET  /health                 - status check
  GET  /v1/models              - OpenAI-style model list
  POST /v1/chat/completions    - OpenAI-style chat (text + image_url, optional stream)
  POST /ocr                    - multipart upload: file=<image or PDF>, prompt=<optional>
  POST /pdf/render             - multipart upload: file=<PDF> -> one image per page (used by the web UI)
  GET  /docs                   - Swagger UI for trying the endpoints

Config via env vars: VLLM_URL, MODEL_ID, HOST, PORT, MAX_PIXELS, DEFAULT_MAX_TOKENS,
MAX_PDF_PAGES, PDF_MAX_DPI, OCR_CONCURRENCY.
"""
import asyncio
import base64
import binascii
import io
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import pypdfium2 as pdfium
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL_ID = os.getenv("MODEL_ID", "nectec/Pathumma-llm-vision-3.0.0-preview")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
# Images larger than this are downscaled before encoding (1 visual token ~= 32x32 px).
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "2000000"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
DEFAULT_OCR_PROMPT = "อ่านข้อความในภาพนี้"
WEB_DIR = Path(__file__).parent / "web"
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "100"))
# PDF pages are rendered to fill MAX_PIXELS, but never above this DPI (small pages).
PDF_MAX_DPI = int(os.getenv("PDF_MAX_DPI", "300"))
# PDF pages sent to vLLM at once (vLLM batches them; extra ones wait in its queue)
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))

# pdfium is not thread-safe; every pdfium call goes through this lock.
pdf_lock = threading.Lock()

app = FastAPI(title="Pathumma Vision OCR", version="1.0")


# ---------- helpers ----------

def load_image(src: Union[str, bytes]) -> Image.Image:
    try:
        if isinstance(src, bytes):
            data = src
        elif src.startswith("data:"):
            data = base64.b64decode(src.split(",", 1)[1])
        elif src.startswith(("http://", "https://")):
            r = requests.get(src, timeout=30)
            r.raise_for_status()
            data = r.content
        else:
            data = base64.b64decode(src, validate=True)
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Unsupported or corrupt image file")
    except (binascii.Error, ValueError, requests.RequestException) as e:
        raise HTTPException(400, f"Cannot load image: {e}")

    w, h = img.size
    if w * h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (w * h))
        img = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.Resampling.BICUBIC)
    return img


def is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip().startswith(b"%PDF")


def pdf_pages(data: bytes):
    """Yield each PDF page as a PIL image sized to the model's pixel budget."""
    with pdf_lock:
        try:
            pdf = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as e:
            raise HTTPException(400, f"Cannot open PDF: {e}")
        n = len(pdf)
    try:
        if n == 0:
            raise HTTPException(400, "PDF has no pages")
        if n > MAX_PDF_PAGES:
            raise HTTPException(400, f"PDF has {n} pages; the limit is {MAX_PDF_PAGES} (set MAX_PDF_PAGES)")
        for i in range(n):
            with pdf_lock:
                page = pdf[i]
                w, h = page.get_size()  # points (1/72 inch)
                scale = min(math.sqrt(MAX_PIXELS / (w * h)), PDF_MAX_DPI / 72)
                img = page.render(scale=scale).to_pil().convert("RGB")
                page.close()
            yield img
    finally:
        with pdf_lock:
            pdf.close()


def clean_ocr_text(text: str) -> str:
    # the model escapes numbered lists Markdown-style ("1\. item"); undo it for plain-text output
    return re.sub(r"^(\s*\d+)\\\.", r"\1.", text, flags=re.MULTILINE)


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def to_openai(messages) -> list:
    """Downscale every image to MAX_PIXELS and inline it as a data URL for vLLM."""
    out = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            parts = []
            for p in content:
                kind = p.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif kind in ("image_url", "image"):
                    url = p.get("image_url") or p.get("image")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if not url:
                        raise HTTPException(400, "image item has no url")
                    parts.append({"type": "image_url", "image_url": {"url": to_data_url(load_image(url))}})
                else:
                    raise HTTPException(400, f"Unsupported content type: {kind}")
            content = parts
        out.append({"role": m.role, "content": content})
    return out


def vllm_post(body: dict, stream: bool = False) -> requests.Response:
    try:
        r = requests.post(f"{VLLM_URL}/v1/chat/completions", json={"model": MODEL_ID, **body},
                          stream=stream, timeout=(10, 600))
    except requests.RequestException as e:
        raise HTTPException(503, f"vLLM is not reachable: {e}")
    if r.status_code != 200:
        try:
            err = r.json()
            msg = err.get("message") or err.get("error", {}).get("message") or r.text
        except ValueError:
            msg = r.text
        r.close()
        raise HTTPException(r.status_code, f"vLLM: {msg}")
    return r


def ocr_image(img: Image.Image, prompt: str, max_tokens: int, repetition_penalty: float) -> dict:
    data = vllm_post({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": to_data_url(img)}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "repetition_penalty": repetition_penalty,
    }).json()
    c = data["choices"][0]
    return {
        "text": clean_ocr_text(c["message"]["content"].strip()),
        "completion_tokens": data["usage"]["completion_tokens"],
        "finish_reason": c["finish_reason"],
        "image_size": list(img.size),
    }


# ---------- API ----------

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict], None] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.0  # greedy by default - best for OCR
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stream: bool = False


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    try:
        requests.get(f"{VLLM_URL}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        raise HTTPException(503, "vLLM is not ready")
    return {"status": "ok", "model": MODEL_ID, "device": "vllm", "max_pixels": MAX_PIXELS}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "nectec"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    body = {
        "messages": to_openai(req.messages),
        "max_tokens": req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
        "stream": req.stream,
    }
    r = vllm_post(body, stream=req.stream)
    if not req.stream:
        return r.json()

    async def relay():
        # Starlette cancels this on client disconnect; closing the upstream socket makes vLLM abort the request
        chunks = r.iter_content(chunk_size=None)
        try:
            while (chunk := await asyncio.to_thread(next, chunks, None)) is not None:
                yield chunk
        finally:
            r.close()

    return StreamingResponse(relay(), media_type="text/event-stream")


@app.post("/ocr")
def ocr(
    file: UploadFile = File(..., description="Image file (jpg/png/webp/...) or PDF"),
    prompt: str = Form(DEFAULT_OCR_PROMPT),
    max_tokens: int = Form(DEFAULT_MAX_TOKENS),
    repetition_penalty: float = Form(1.0),
):
    data = file.file.read()
    t0 = time.time()

    if is_pdf(data):
        # ponytail: renders every page up front (~6 MB each, MAX_PDF_PAGES caps it); stream pages if that bites
        with ThreadPoolExecutor(OCR_CONCURRENCY) as ex:
            results = list(ex.map(lambda img: ocr_image(img, prompt, max_tokens, repetition_penalty), pdf_pages(data)))
        pages = [{"page": i, **res} for i, res in enumerate(results, start=1)]
        return {
            "text": "\n\n".join(p["text"] for p in pages),
            "pages": pages,
            "elapsed_s": round(time.time() - t0, 2),
        }

    return {**ocr_image(load_image(data), prompt, max_tokens, repetition_penalty),
            "elapsed_s": round(time.time() - t0, 2)}


@app.post("/pdf/render")
def pdf_render(file: UploadFile = File(..., description="PDF file")):
    data = file.file.read()
    if not is_pdf(data):
        raise HTTPException(400, "Not a PDF file")
    pages = [
        {"page": i, "width": img.width, "height": img.height, "image": to_data_url(img)}
        for i, img in enumerate(pdf_pages(data), start=1)
    ]
    return {"page_count": len(pages), "pages": pages}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
