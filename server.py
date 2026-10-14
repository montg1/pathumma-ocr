"""OpenAI-compatible inference server for nectec/Pathumma-llm-vision-3.0.0-preview.

Endpoints
  GET  /                       - web UI: upload images and OCR them
  GET  /health                 - status check
  GET  /v1/models              - OpenAI-style model list
  POST /v1/chat/completions    - OpenAI-style chat (text + image_url, optional stream)
  POST /ocr                    - multipart upload: file=<image or PDF>, prompt=<optional>
  POST /pdf/render             - multipart upload: file=<PDF> -> one image per page (used by the web UI)
  GET  /docs                   - Swagger UI for trying the endpoints

Config via env vars: MODEL_ID, HOST, PORT, MAX_PIXELS, DEFAULT_MAX_TOKENS, WARMUP (1/0),
MAX_PDF_PAGES, PDF_MAX_DPI.
"""
import asyncio
import base64
import binascii
import io
import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Union

import pypdfium2 as pdfium
import requests
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_ID} on {device} ...", flush=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device,
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
eos_ids = model.generation_config.eos_token_id
eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
# One GPU, one generation at a time.
gpu_lock = threading.Lock()
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


def to_template(messages) -> tuple:
    """Convert OpenAI-style messages to the model's chat format + list of PIL images."""
    conv, images = [], []
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
                    images.append(load_image(url))
                    parts.append({"type": "image"})
                else:
                    raise HTTPException(400, f"Unsupported content type: {kind}")
            content = parts
        conv.append({"role": m.role, "content": content})
    return conv, images


def build_inputs(conv, images):
    prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images or None, return_tensors="pt")
    return inputs.to(model.device)


def gen_kwargs(max_tokens: int, temperature: float, top_p: float, repetition_penalty: float) -> dict:
    kw = dict(max_new_tokens=max_tokens, use_cache=True, repetition_penalty=repetition_penalty)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw.update(do_sample=False)
    return kw


def finish_reason(new_tokens, max_tokens: int) -> str:
    if len(new_tokens) >= max_tokens and int(new_tokens[-1]) not in eos_ids:
        return "length"
    return "stop"


def generate(inputs, kw) -> tuple:
    with gpu_lock, torch.inference_mode():
        out = model.generate(**inputs, **kw)
    new = out[0, inputs.input_ids.shape[1]:]
    text = processor.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text.strip(), len(new), finish_reason(new, kw["max_new_tokens"])


class _Cancelled(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


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
    return {"status": "ok", "model": MODEL_ID, "device": device, "max_pixels": MAX_PIXELS}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "nectec"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    max_tokens = req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS
    conv, images = to_template(req.messages)
    inputs = build_inputs(conv, images)
    kw = gen_kwargs(max_tokens, req.temperature, req.top_p, req.repetition_penalty)
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(_stream(inputs, kw, req_id, created), media_type="text/event-stream")

    text, n_out, finish = generate(inputs, kw)
    n_in = inputs.input_ids.shape[1]
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
    }


async def _stream(inputs, kw, req_id, created):
    def chunk(delta: dict, finish: Optional[str] = None) -> str:
        body = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    cancelled = threading.Event()
    result = {}

    # The worker owns the GPU lock, so a client that disconnects mid-stream
    # can never leave the lock held by a suspended generator.
    def run():
        try:
            with gpu_lock, torch.inference_mode():
                result["out"] = model.generate(
                    **inputs, **kw, streamer=streamer, stopping_criteria=StoppingCriteriaList([_Cancelled(cancelled)])
                )
        except Exception as e:  # unblock the consumer if generation fails
            result["error"] = e
            streamer.end()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    pieces = iter(streamer)
    done = object()
    try:
        yield chunk({"role": "assistant", "content": ""})
        while (piece := await asyncio.to_thread(next, pieces, done)) is not done:
            if piece:
                yield chunk({"content": piece})
        await asyncio.to_thread(worker.join)
        if "error" in result:
            yield chunk({}, "error")
        else:
            new = result["out"][0, inputs.input_ids.shape[1]:]
            yield chunk({}, finish_reason(new, kw["max_new_tokens"]))
        yield "data: [DONE]\n\n"
    finally:
        # client disconnected (or finished): stop generating at the next token
        cancelled.set()


@app.post("/ocr")
def ocr(
    file: UploadFile = File(..., description="Image file (jpg/png/webp/...) or PDF"),
    prompt: str = Form(DEFAULT_OCR_PROMPT),
    max_tokens: int = Form(DEFAULT_MAX_TOKENS),
    repetition_penalty: float = Form(1.0),
):
    data = file.file.read()
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    kw = gen_kwargs(max_tokens, 0, 1.0, repetition_penalty)
    t0 = time.time()

    if is_pdf(data):
        pages = []
        for i, img in enumerate(pdf_pages(data), start=1):
            text, n_out, finish = generate(build_inputs(conv, [img]), kw)
            pages.append({"page": i, "text": clean_ocr_text(text), "completion_tokens": n_out, "finish_reason": finish,
                          "image_size": list(img.size)})
        return {
            "text": "\n\n".join(p["text"] for p in pages),
            "pages": pages,
            "elapsed_s": round(time.time() - t0, 2),
        }

    img = load_image(data)
    text, n_out, finish = generate(build_inputs(conv, [img]), kw)
    return {
        "text": clean_ocr_text(text),
        "completion_tokens": n_out,
        "finish_reason": finish,
        "image_size": list(img.size),
        "elapsed_s": round(time.time() - t0, 2),
    }


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


def warmup():
    """Compile/autotune the Triton kernels before the first real request."""
    t0 = time.time()
    img = Image.new("RGB", (512, 256), "white")
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DEFAULT_OCR_PROMPT}]}]
    generate(build_inputs(conv, [img]), gen_kwargs(8, 0, 1.0, 1.0))
    print(f"Warmup done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    if os.getenv("WARMUP", "1") == "1":
        warmup()
    uvicorn.run(app, host=HOST, port=PORT)
