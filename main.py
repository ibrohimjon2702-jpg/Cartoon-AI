import os
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from huggingface_hub import InferenceClient

app = FastAPI(title="Cartoon AI Video API")

# Productionda CORS_ORIGINS ni o'zingizning sayt domeningiz bilan cheklang.
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in origins if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
TEXT_MODEL = os.getenv("TEXT_VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
IMAGE_MODEL = os.getenv("IMAGE_VIDEO_MODEL", "Wan-AI/Wan2.2-I2V-A14B")
PROVIDER = os.getenv("HF_PROVIDER", "fal-ai")


def get_client():
    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Serverda HF_TOKEN sozlanmagan. Hugging Face tokenini server Environment Variables bo'limiga kiriting.",
        )
    return InferenceClient(provider=PROVIDER, api_key=HF_TOKEN, timeout=300)


@app.get("/")
def health():
    return {
        "app": "Cartoon AI Video API",
        "status": "ok",
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "note": "Video yaratish uchun provider hisobida ruxsat va kredit kerak bo'lishi mumkin.",
    }


@app.post("/api/text-to-video")
def text_to_video(prompt: str = Form(...)):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt bo'sh bo'lmasin.")
    if len(prompt) > 3000:
        raise HTTPException(status_code=400, detail="Prompt juda uzun (maksimum 3000 belgi).")
    client = get_client()
    try:
        video = client.text_to_video(prompt, model=TEXT_MODEL)
    except Exception as exc:
        # Foydalanuvchiga token yoki maxfiy server tafsilotlarini oshkor qilmaymiz.
        raise HTTPException(
            status_code=502,
            detail="Video yaratilmadi. Model/provider mavjudligini, HF token ruxsatini va kredit qoldig'ini tekshiring. Tafsilot: " + str(exc)[:400],
        )
    return Response(
        content=bytes(video),
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="cartoon-ai-text.mp4"'},
    )


@app.post("/api/image-to-video")
async def image_to_video(
    prompt: str = Form(...),
    image: UploadFile = File(...),
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Harakat promptini kiriting.")
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="JPG, PNG yoki WEBP rasm tanlang.")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Rasm bo'sh.")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Rasm hajmi 10 MB dan kichik bo'lsin.")
    client = get_client()
    try:
        video = client.image_to_video(data, model=IMAGE_MODEL, prompt=prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Rasmli video yaratilmadi. Model/provider mavjudligini, HF token ruxsatini va kredit qoldig'ini tekshiring. Tafsilot: " + str(exc)[:400],
        )
    return Response(
        content=bytes(video),
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="cartoon-ai-image.mp4"'},
    )
