# Pathumma OCR

บริการ OCR ภาษาไทยจากรูปภาพและ PDF โดยใช้โมเดล [nectec/Pathumma-llm-vision-3.0.0-preview](https://huggingface.co/nectec/Pathumma-llm-vision-3.0.0-preview) (Qwen3.5-2B ที่ NECTEC ฝึกเพิ่มสำหรับ OCR ภาษาไทย) รันบน GPU ในเครื่อง

- **หน้าเว็บ** อัปโหลดรูปหรือ PDF แล้วอ่านข้อความ แสดงผลแบบสตรีม คัดลอก/ดาวน์โหลดเป็น .txt ได้
- **API** แบบ OpenAI-compatible (`/v1/chat/completions`) และ endpoint อัปโหลดไฟล์ตรง ๆ (`/ocr`)
- **PDF** แยกเป็นรายหน้าอัตโนมัติ
- รันได้ทั้งแบบ **Docker Compose** และแบบ **Python ตรงบน Windows**

## ความต้องการของระบบ

| | ขั้นต่ำ |
|---|---|
| GPU | NVIDIA VRAM 8 GB (ทดสอบกับ RTX 3070 8 GB, ตัวโมเดลใช้ประมาณ 4.8 GB) |
| Driver | NVIDIA driver ที่รองรับ CUDA 12.8 ขึ้นไป |
| ดิสก์ | ~12 GB สำหรับ Docker image + 4.4 GB สำหรับ weights ของโมเดล |
| แบบ Docker | Docker Desktop (WSL2 backend) หรือ Docker บน Linux + NVIDIA Container Toolkit |
| แบบไม่ใช้ Docker | Python 3.12 + [uv](https://docs.astral.sh/uv/) |

## เริ่มใช้งานด้วย Docker Compose (แนะนำ)

```bash
docker compose up -d --build
```

ครั้งแรกจะใช้เวลานาน: build image (~5–10 นาที ขึ้นกับความเร็วเน็ต) และดาวน์โหลดโมเดล 4.4 GB ตอนเริ่ม container ดูความคืบหน้าได้ด้วย

```bash
docker compose logs -f
```

เมื่อเห็น `Uvicorn running on http://0.0.0.0:8000` ให้เปิด **http://127.0.0.1:8000**

ครั้งต่อ ๆ ไปจะพร้อมใช้ในประมาณ 30 วินาที เพราะโมเดลและ kernel ที่ compile แล้วเก็บไว้ใน volume `pathumma-data`

| คำสั่ง | ใช้ทำอะไร |
|---|---|
| `docker compose up -d` | เริ่ม service |
| `docker compose up -d --build` | build ใหม่หลังแก้โค้ด แล้วเริ่ม service |
| `docker compose logs -f` | ดู log |
| `docker compose down` | หยุดและลบ container (โมเดลใน volume ยังอยู่) |
| `docker compose down -v` | หยุดและ **ลบ volume ด้วย** ครั้งหน้าจะดาวน์โหลดโมเดลใหม่ |

container ตั้งเป็น `restart: unless-stopped` จึงเริ่มเองทุกครั้งที่เปิด Docker Desktop จนกว่าจะสั่ง `docker compose down`

### ตั้งค่า (ไฟล์ `.env` ข้าง `docker-compose.yml`)

```env
OCR_BIND=127.0.0.1     # 0.0.0.0 = เปิดให้เครื่องอื่นใน LAN ใช้ได้
OCR_PORT=8000          # พอร์ตฝั่งเครื่อง host
MAX_PIXELS=2000000
DEFAULT_MAX_TOKENS=2048
MAX_PDF_PAGES=100
```

## รันแบบไม่ใช้ Docker (Windows)

สร้าง environment ไว้นอกโฟลเดอร์ OneDrive เพื่อไม่ให้ sync ไฟล์ torch หลาย GB

```powershell
uv venv --python 3.12 "$env:USERPROFILE\.venvs\pathumma-vision"
$py = "$env:USERPROFILE\.venvs\pathumma-vision\Scripts\python.exe"
uv pip install --python $py torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $py -r requirements.txt
```

เริ่ม server

```powershell
powershell -ExecutionPolicy Bypass -File start_server.ps1
```

ครั้งแรกจะดาวน์โหลดโมเดลไปไว้ที่ `%USERPROFILE%\.cache\huggingface`

> รันแบบ Docker กับแบบนี้พร้อมกันไม่ได้ เพราะใช้พอร์ตเดียวกันและ VRAM 8 GB ไม่พอสำหรับโมเดล 2 ชุด ให้ `docker compose down` ก่อน

## หน้าเว็บ

เปิด http://127.0.0.1:8000

1. ลากรูปหรือ PDF มาวาง คลิกเพื่อเลือกไฟล์ หรือกด Ctrl+V วางรูปจากคลิปบอร์ด (เลือกได้หลายไฟล์ PDF จะแยกเป็นทีละหน้า)
2. กด **อ่านข้อความ** สำหรับรูปที่เลือก หรือ **อ่านทุกรูป** เพื่ออ่านทุกหน้าตามลำดับ กด **หยุด** ได้ระหว่างอ่าน
3. แก้ไขผลลัพธ์ได้ในช่องข้อความ แล้ว **คัดลอก**, **ดาวน์โหลด .txt** หรือ **ดาวน์โหลดรวมทุกหน้า**

คำสั่งสำเร็จรูปมี 2 แบบ: *อ่านข้อความ* และ *แปลงเป็น Markdown* ปรับ max tokens และ repetition penalty ได้ใน **ตั้งค่าเพิ่มเติม** (ถ้าผลลัพธ์วนซ้ำ ลองตั้ง repetition penalty เป็น 1.05)

## API

ดูและทดลองทุก endpoint ได้ที่ http://127.0.0.1:8000/docs

| Method | Path | คำอธิบาย |
|---|---|---|
| GET | `/` | หน้าเว็บ |
| GET | `/health` | สถานะ server |
| GET | `/v1/models` | รายชื่อโมเดล (รูปแบบ OpenAI) |
| POST | `/v1/chat/completions` | แชตแบบ OpenAI รองรับ `image_url` (data URL / http URL / base64) และ `stream` |
| POST | `/ocr` | อัปโหลดรูปหรือ PDF (multipart) ได้ข้อความกลับมา |
| POST | `/pdf/render` | แปลง PDF เป็นรูปทีละหน้า (หน้าเว็บใช้ endpoint นี้) |

### `/ocr` อัปโหลดไฟล์

```bash
curl -F "file=@receipt.jpg" http://127.0.0.1:8000/ocr
```

```json
{"text": "ใบเสร็จรับเงิน เลขที่ 0012345\n...", "completion_tokens": 95, "finish_reason": "stop", "image_size": [900, 420], "elapsed_s": 5.24}
```

ส่ง PDF ได้เหมือนกัน จะได้ข้อความรวม (`text`) และผลรายหน้า (`pages`)

```bash
curl -F "file=@document.pdf" http://127.0.0.1:8000/ocr
```

ฟิลด์เพิ่มเติม (ไม่บังคับ): `prompt` (ค่าเริ่มต้น `อ่านข้อความในภาพนี้`), `max_tokens`, `repetition_penalty`

> บน Windows PowerShell ให้ใช้ `curl.exe` แทน `curl`

### `/v1/chat/completions` แบบ OpenAI

```python
import base64, requests

b64 = base64.b64encode(open("receipt.jpg", "rb").read()).decode()
r = requests.post("http://127.0.0.1:8000/v1/chat/completions", json={
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "อ่านข้อความในภาพนี้"},
        ],
    }],
    "max_tokens": 2048,
})
print(r.json()["choices"][0]["message"]["content"])
```

ค่าเริ่มต้น `temperature` เป็น 0 (greedy) ซึ่งเหมาะกับ OCR ส่ง `"stream": true` เพื่อรับผลแบบ Server-Sent Events

ทดสอบเร็ว ๆ ด้วยสคริปต์ที่มีให้

```bash
python test_client.py receipt.jpg
```

## การตั้งค่า (environment variables)

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `HOST` | `127.0.0.1` (Docker: `0.0.0.0`) | address ที่ server ฟัง |
| `PORT` | `8000` | พอร์ตของ server |
| `MAX_PIXELS` | `2000000` | รูปที่ใหญ่กว่านี้จะถูกย่อก่อนส่งเข้าโมเดล เพิ่มได้ถ้าตัวอักษรเล็กอ่านไม่ออก แต่จะใช้ VRAM และเวลามากขึ้น |
| `DEFAULT_MAX_TOKENS` | `2048` | ความยาวผลลัพธ์สูงสุดเมื่อไม่ได้ระบุ |
| `MAX_PDF_PAGES` | `100` | จำนวนหน้า PDF สูงสุดต่อไฟล์ |
| `PDF_MAX_DPI` | `300` | DPI สูงสุดตอนแปลงหน้า PDF (หน้าปกติจะถูกจำกัดด้วย `MAX_PIXELS` ก่อน เช่น A4 ≈ 144 DPI) |
| `WARMUP` | `1` | compile kernel ตอนเริ่ม server เพื่อให้ request แรกไม่ช้า (`0` = ปิด) |
| `MODEL_ID` | `nectec/Pathumma-llm-vision-3.0.0-preview` | โมเดลบน Hugging Face |

## ประสิทธิภาพ (RTX 3070 8 GB)

| งาน | เวลา |
|---|---|
| ใบเสร็จ 6 บรรทัด (95 tokens) | ~5 วินาที |
| PDF หน้า A4 | ~3–5 วินาทีต่อหน้า |
| เริ่ม server (โมเดลอยู่ในเครื่องแล้ว) | ~30 วินาที |

ความเร็วนี้ได้จาก `flash-linear-attention` ถ้าไม่ได้ติดตั้งจะช้าลงประมาณ 1.7 เท่า

## ข้อจำกัด

- โมเดลนี้ฝึกมาเพื่อ **OCR โดยเฉพาะ** คำสั่งแนวถามตอบ สรุป หรือดึงเป็น JSON มักได้ข้อความทั้งหมดในภาพกลับมาแทน
- รับได้ทีละ request (มี 1 GPU) request อื่นจะรอคิว
- **ไม่มีระบบยืนยันตัวตน** ถ้าเปิดให้เครื่องอื่นใช้ (`OCR_BIND=0.0.0.0` หรือ `HOST=0.0.0.0`) ควรทำเฉพาะในเครือข่ายที่ไว้ใจได้
- ลายมือเขียนยังอ่านได้ไม่ดี (ThaiOCRBench ส่วน handwritten = 0.33 ตาม model card)
- ข้อความสั้นบรรทัดเดียวบางครั้งโมเดลตัดช่องว่างระหว่างคำทิ้ง
- รองรับรูปที่ Pillow เปิดได้ (JPG, PNG, WEBP, BMP, TIFF ฯลฯ) ยังไม่รองรับ HEIC

## โครงสร้างโปรเจกต์

```
server.py            FastAPI server (โหลดโมเดล, API, แปลง PDF)
web/index.html       หน้าเว็บ (ไฟล์เดียว ไม่พึ่ง CDN ใช้งาน offline ได้)
requirements.txt     dependencies (ยกเว้น torch ที่ติดตั้งจาก index ของ CUDA)
Dockerfile           image สำหรับ Docker
docker-compose.yml   service + GPU + volume
start_server.ps1     เริ่ม server แบบไม่ใช้ Docker บน Windows
test_client.py       สคริปต์ทดสอบ API
```

## เครดิต

โมเดล [Pathumma-llm-vision-3.0.0-preview](https://huggingface.co/nectec/Pathumma-llm-vision-3.0.0-preview) พัฒนาโดย NECTEC เผยแพร่ภายใต้สัญญาอนุญาต Apache-2.0 ต่อยอดจาก Qwen3.5-2B ของทีม Qwen
