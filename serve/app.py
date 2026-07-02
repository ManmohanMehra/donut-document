import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from inference import load_model, run_inference
from schemas import SUPPORTED_CARD_TYPES

# ── Global model objects (loaded once at startup) ──────────────────────────
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["model"], model_state["processor"], model_state["device"] = load_model()
    print("Model loaded and ready.")
    yield
    model_state.clear()


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Donut Document Parser",
    description="Extract structured data from identity documents using a fine-tuned Donut model.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model_state), "supported_card_types": SUPPORTED_CARD_TYPES}


@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    card_type: str = Form("indian_passport"),
):
    """
    Upload a document image (JPEG/PNG) and receive extracted fields as JSON.

    Form fields:
        file      — image file (JPEG or PNG)
        card_type — one of the supported card types (default: indian_passport)
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    if card_type not in SUPPORTED_CARD_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported card_type '{card_type}'. Supported: {SUPPORTED_CARD_TYPES}",
        )

    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_inference(
            tmp_path,
            card_type=card_type,
            model=model_state["model"],
            processor=model_state["processor"],
            device=model_state["device"],
        )
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)


# ── Legacy endpoint (backward compat) ─────────────────────────────────────
@app.post("/process-passport")
async def process_passport(file: UploadFile = File(...)):
    """Deprecated: use /process-document with card_type=indian_passport."""
    return await process_document(file=file, card_type="indian_passport")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
