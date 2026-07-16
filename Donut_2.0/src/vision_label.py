"""
vision_label.py — Gemini Vision pre-labeler for any card type in schemas.py

Uses Google Gemini (free via Google AI Studio) to extract fields from document
images. Output is a metadata_suggested.jsonl that annotators review/correct in
Label Studio instead of typing from scratch.

Cost: FREE for up to 1,500 images/day using the free AI Studio API key.
      If using GCP credits (Vertex AI), set USE_VERTEX=1 (see below).

Setup:
    pip install google-genai pillow tqdm
    (the older google-generativeai package is deprecated — no more updates/fixes)

    Option A — Free AI Studio key (recommended for 152 images):
        Get key at: https://aistudio.google.com/app/apikey
        export GOOGLE_API_KEY=AIza...

    Option B — GCP credits via Vertex AI:
        export USE_VERTEX=1
        export GCP_PROJECT=your-project-id
        export GCP_LOCATION=us-central1
        gcloud auth application-default login
        NOTE: the new google-genai SDK's Vertex-mode constructor arg has
        moved around across SDK versions (vertexai=True vs enterprise=True
        in different releases) — if this path errors on your installed
        version, check https://github.com/googleapis/python-genai for the
        current signature. Option A (AI Studio) doesn't have this risk.

Usage:
    python src/vision_label.py data/are_fed_card/images  data/are_fed_card/metadata_suggested.jsonl  are_fed_card
    python src/vision_label.py data/cod/images           data/cod/metadata_suggested.jsonl           cod_passport
    python src/vision_label.py data/zwe/images           data/zwe/metadata_suggested.jsonl           zwe_passport
"""

import os
import json
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm

from schemas import SCHEMAS, SUPPORTED_CARD_TYPES
from mrz_validator import validate_mrz

try:
    from google import genai
    from google.genai import types
    from PIL import Image
except ImportError:
    print("Required packages not found. Run:")
    print("  pip install google-genai pillow tqdm")
    sys.exit(1)

MODEL_NAME = "gemini-3.5-flash"  # gemini-2.0-flash was deprecated 2026-06-01 (free-tier quota drops to 0)


def _setup_client():
    use_vertex = os.environ.get("USE_VERTEX", "").strip() == "1"

    if use_vertex:
        project = os.environ.get("GCP_PROJECT")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        if not project:
            print("Set GCP_PROJECT environment variable for Vertex AI.")
            sys.exit(1)
        print(f"Using Vertex AI (project={project}, location={location})")
        client = genai.Client(vertexai=True, project=project, location=location)
        return client, "vertex"
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Set GOOGLE_API_KEY environment variable.")
            print("Get a free key at: https://aistudio.google.com/app/apikey")
            sys.exit(1)
        print(f"Using Google AI Studio (free tier, model={MODEL_NAME})")
        client = genai.Client(api_key=api_key)
        return client, "aistudio"


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


def _extract_fields(client, image_path: str, card_type: str) -> dict:
    prompt = _build_prompt(card_type)
    img = Image.open(image_path)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, img],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# Free-tier RPM isn't published per-model and varies by account tier (see
# https://aistudio.google.com/rate-limit for your actual numbers) — rather
# than guess a fixed request interval and risk being wrong in either
# direction, this reacts to real transient failures instead:
#   429 RESOURCE_EXHAUSTED — rate limit; Google's error includes a retryDelay, used directly
#   503 UNAVAILABLE / 500 INTERNAL — transient server overload; no retryDelay given,
#     so this backs off exponentially (15s, 30s, 60s, ...)
# Anything else (bad request, auth failure, malformed JSON) is not transient
# and is raised immediately rather than wasting retries on it.
MAX_TRANSIENT_RETRIES = 5


def _extract_fields_with_retry(client, image_path: str, card_type: str) -> dict:
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            return _extract_fields(client, image_path, card_type)
        except Exception as e:
            msg = str(e)
            is_transient = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL"))
            if not is_transient or attempt == MAX_TRANSIENT_RETRIES:
                raise
            m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", msg)
            delay = float(m.group(1)) + 1.0 if m else min(60.0, 15.0 * (2 ** attempt))
            time.sleep(delay)


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

    client, backend = _setup_client()

    input_path = Path(input_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(f for f in os.listdir(input_path) if Path(f).suffix.lower() in exts)

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Gemini labeling {len(images)} images as '{card_type}' ...")
    if backend == "aistudio":
        print("Free tier RPM varies by account — see https://aistudio.google.com/rate-limit. "
              "Hitting a real 429 backs off automatically using Google's suggested retry delay.\n")
    else:
        print()

    errors = []
    mrz_flags = []

    with open(output_file, "w", encoding="utf-8") as out:
        for img_name in tqdm(images):
            img_path = str(input_path / img_name)
            try:
                fields = _extract_fields_with_retry(client, img_path, card_type)

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
