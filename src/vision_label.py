"""
vision_label.py — Gemini Vision pre-labeler for any card type in schemas.py

Uses Google Gemini (free via Google AI Studio) to extract fields from document
images. Output is a metadata_suggested.jsonl that annotators review/correct in
Label Studio instead of typing from scratch.

Cost: FREE for up to 1,500 images/day using the free AI Studio API key.
      If using GCP credits (Vertex AI), set USE_VERTEX=1 (see below).

Setup:
    pip install google-generativeai pillow tqdm

    Option A — Free AI Studio key (recommended for 152 images):
        Get key at: https://aistudio.google.com/app/apikey
        export GOOGLE_API_KEY=AIza...

    Option B — GCP credits via Vertex AI:
        export USE_VERTEX=1
        export GCP_PROJECT=your-project-id
        export GCP_LOCATION=us-central1
        gcloud auth application-default login

Usage:
    python src/vision_label.py data/are_fed_card/images  data/are_fed_card/metadata_suggested.jsonl  are_fed_card
    python src/vision_label.py data/cod/images           data/cod/metadata_suggested.jsonl           cod_passport
    python src/vision_label.py data/zwe/images           data/zwe/metadata_suggested.jsonl           zwe_passport
"""

import os
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

from schemas import SCHEMAS, SUPPORTED_CARD_TYPES
from mrz_validator import validate_mrz

try:
    import google.generativeai as genai
    from PIL import Image
except ImportError:
    print("Required packages not found. Run:")
    print("  pip install google-generativeai pillow tqdm")
    sys.exit(1)

MODEL_NAME = "gemini-2.0-flash"


def _setup_client():
    use_vertex = os.environ.get("USE_VERTEX", "").strip() == "1"

    if use_vertex:
        project = os.environ.get("GCP_PROJECT")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        if not project:
            print("Set GCP_PROJECT environment variable for Vertex AI.")
            sys.exit(1)
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=project, location=location)
        print(f"Using Vertex AI (project={project}, location={location})")
        return GenerativeModel(MODEL_NAME), "vertex"
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Set GOOGLE_API_KEY environment variable.")
            print("Get a free key at: https://aistudio.google.com/app/apikey")
            sys.exit(1)
        genai.configure(api_key=api_key)
        print(f"Using Google AI Studio (free tier, model={MODEL_NAME})")
        return genai.GenerativeModel(MODEL_NAME), "aistudio"


def _build_prompt(card_type: str) -> str:
    schema = SCHEMAS[card_type]
    fields = [k for k in schema if k != "card_type"]
    field_list = "\n".join(f"  - {f}" for f in fields)
    return f"""You are a KYC document parser. Extract fields from this {card_type.replace("_", " ")} image.

Fields to extract:
{field_list}

Rules:
- Return ONLY a valid JSON object with these exact field names as keys.
- Use null for any field that is not visible, not present, or illegible.
- Date fields: DD/MM/YYYY format.
- MRZ lines: copy EXACTLY including '<' fill characters. Each line must be 44 characters.
- Names: uppercase, as printed on the document.
- Do not invent or guess values — null is better than a wrong value."""


def _extract_fields(model, image_path: str, card_type: str, backend: str) -> dict:
    prompt = _build_prompt(card_type)
    img = Image.open(image_path)

    if backend == "vertex":
        from vertexai.generative_models import GenerationConfig
        response = model.generate_content(
            [prompt, img],
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
    else:
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            ),
        )

    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _mrz_flag(fields: dict) -> str | None:
    m1 = fields.get("mrz_line1") or ""
    m2 = fields.get("mrz_line2") or ""
    if not m1 or not m2:
        return None
    result = validate_mrz(m1.strip(), m2.strip())
    if not result.get("valid"):
        return f"MRZ checksum FAIL: {result}"
    return None


def main(input_dir: str, output_file: str, card_type: str):
    if card_type not in SUPPORTED_CARD_TYPES:
        print(f"Unknown card_type '{card_type}'. Supported: {SUPPORTED_CARD_TYPES}")
        sys.exit(1)

    model, backend = _setup_client()

    input_path = Path(input_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(f for f in os.listdir(input_path) if Path(f).suffix.lower() in exts)

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Gemini labeling {len(images)} images as '{card_type}' ...")
    if backend == "aistudio":
        print(f"Free tier: 15 req/min — estimated time ~{len(images) // 15 + 1} min\n")
    else:
        print()

    errors = []
    mrz_flags = []
    last_request_time = 0.0

    with open(output_file, "w", encoding="utf-8") as out:
        for img_name in tqdm(images):
            img_path = str(input_path / img_name)
            try:
                # Rate limit for free tier: 15 req/min = 4s between requests
                if backend == "aistudio":
                    elapsed = time.time() - last_request_time
                    if elapsed < 4.1:
                        time.sleep(4.1 - elapsed)

                fields = _extract_fields(model, img_path, card_type, backend)
                last_request_time = time.time()

                fields.pop("card_type", None)

                flag = _mrz_flag(fields)
                if flag:
                    mrz_flags.append((img_name, flag))

                record = {
                    "file_name": img_name,
                    "ground_truth": {"card_type": card_type, **fields},
                }
                if flag:
                    record["_review_flag"] = flag

                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                errors.append((img_name, str(e)))
                print(f"\nError on {img_name}: {e}")
                last_request_time = time.time()

    print(f"\nDone. {len(images) - len(errors)}/{len(images)} images labeled.")
    if mrz_flags:
        print(f"\n{len(mrz_flags)} images flagged for MRZ review:")
        for name, msg in mrz_flags:
            print(f"  {name}: {msg}")
    if errors:
        print(f"\n{len(errors)} errors (skipped):")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print(f"\nOutput: {output_file}")
    print("Next: import into Label Studio, review flags, correct errors, then run convert_labelstudio.py")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
