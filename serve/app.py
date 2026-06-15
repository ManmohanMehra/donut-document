import sys
import os
# Allow running from the serve/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from inference import load_model, run_inference

# ── Global model objects (loaded once at startup) ──────────────────────────
model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once when the server starts."""
    model_state["model"], model_state["processor"], model_state["device"] = load_model()
    print("✅ Model loaded and ready.")
    yield
    model_state.clear()


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Donut Passport Parser",
    description="Extract structured data from Indian passport images using a fine-tuned Donut model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model_state)}


@app.post("/process-passport")
async def process_passport(file: UploadFile = File(...)):
    """
    Upload a passport image (JPEG/PNG) and receive extracted fields as JSON.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_inference(
            tmp_path,
            model=model_state["model"],
            processor=model_state["processor"],
            device=model_state["device"],
        )
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
