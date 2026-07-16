"""
classify_gemini.py — Second opinion on borderline-similarity candidates.

filter_similarity.py already dropped anything clearly not card-like and kept
anything clearly card-like; this handles the middle band by asking Gemini
"is this an ID/passport-type document, and if so which type" — the same
free-tier pattern already used in Donut_2.0/src/vision_label.py.

Usage:
    export GOOGLE_API_KEY=AIza...
    python src/classify_gemini.py output/filtered/borderline output/filtered/borderline.jsonl output/classified.jsonl
"""
import json
import os
import sys
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Run: pip install google-genai   (the old google-generativeai package is deprecated)")
    sys.exit(1)

MODEL_NAME = "gemini-3.5-flash"  # gemini-2.0-flash was deprecated 2026-06-01 (free-tier quota drops to 0)

# Extend this with whatever card types Donut_2.0/src/schemas.py currently supports.
KNOWN_CARD_TYPES = [
    "indian_passport", "foreign_passport", "are_fed_card", "cod_passport", "zwe_passport",
]

PROMPT = f"""You are sorting candidate images collected for a KYC document-parsing
dataset. Look at this image and answer:

1. Is this a photo/scan of an official ID card, passport, or similar
   government-issued identity document? (not a mockup/template with no real
   layout, not an unrelated object)
2. If yes, which type is it? Prefer one of: {KNOWN_CARD_TYPES}. If it's a real
   document type not in that list, name it in your own words (e.g.
   "kenya_national_id").
3. Rough confidence: high / medium / low.

Return ONLY a JSON object: {{"is_id_document": bool, "card_type": string or null, "confidence": string}}"""


def _setup():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GOOGLE_API_KEY environment variable. Get a free key at https://aistudio.google.com/app/apikey")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def main(images_dir: str, manifest_file: str, out_file: str):
    from PIL import Image

    images_path = Path(images_dir)
    with open(manifest_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    client = _setup()
    last_request = 0.0

    with open(out_file, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            img_path = images_path / rec["file_name"]
            if not img_path.exists():
                continue
            try:
                elapsed = time.time() - last_request
                if elapsed < 4.1:
                    time.sleep(4.1 - elapsed)

                img = Image.open(img_path)
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[PROMPT, img],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                last_request = time.time()
                classification = json.loads(response.text.strip())
                out.write(json.dumps({**rec, **classification}) + "\n")
            except Exception as e:
                print(f"  error on {rec['file_name']}: {e}")
                last_request = time.time()

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(records)}")

    print(f"\nClassified -> {out_file}")
    print("Next: route each image into Donut_2.0/data/<card_type>/images/ by its predicted card_type, "
          "then run the normal Phase 1 verification pipeline on it — Gemini's guess here is not ground truth.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
