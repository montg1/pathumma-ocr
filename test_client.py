"""Quick test: python test_client.py <image> [server_url]"""
import base64
import sys

import requests

image = sys.argv[1]
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

print("== /ocr ==")
with open(image, "rb") as f:
    r = requests.post(f"{url}/ocr", files={"file": f}, timeout=600)
r.raise_for_status()
print(r.json())

print("\n== /v1/chat/completions ==")
b64 = base64.b64encode(open(image, "rb").read()).decode()
r = requests.post(
    f"{url}/v1/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "อ่านข้อความในภาพนี้"},
                ],
            }
        ],
        "max_tokens": 1024,
    },
    timeout=600,
)
r.raise_for_status()
data = r.json()
print(data["choices"][0]["message"]["content"])
print(data["usage"])
