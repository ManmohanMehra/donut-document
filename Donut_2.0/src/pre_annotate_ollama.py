"""
pre_annotate_ollama.py — Local pre-annotation via Ollama running MiniCPM-V,
as a genuinely-local alternative to both the Gemini API (rate-limited,
deprioritizes free tier) and pre_annotate_minicpm.py (raw transformers,
unconfirmed Apple Silicon support).

Ollama has confirmed native Apple Silicon acceleration (MLX-backed as of
2026) and a stable, documented REST API — this actually runs on your Mac,
not just a GPU server. No API cost, no rate limit, fully offline once the
model is pulled. Output format matches vision_label.py /
pre_annotate_minicpm.py exactly, so Label Studio import and
convert_labelstudio.py don't care which backend produced the suggestions.

Setup (one-time):
    brew install ollama          # or https://ollama.com/download
    ollama serve                 # starts the local server on :11434 — leave running in another terminal
    ollama pull minicpm-v4.6     # ~4-5GB download

    NOTE: the bare tag "minicpm-v" resolves to the older MiniCPM-V-2.6 on
    Ollama's library, not the current 4.6 — this script pins "minicpm-v4.6"
    explicitly.

Usage — same CLI shape as vision_label.py:
    python src/pre_annotate_ollama.py data/ZWE  data/ZWE/metadata_suggested.jsonl  zwe_passport

Expect this to be slow per-image on a laptop (no dedicated GPU) — that's
normal. It has no rate limit and no cost, so leaving it running overnight
for the full ~730-image set is a reasonable tradeoff.
"""
import base64
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

try:
    import requests
except ImportError:
    print("Run: pip install requests")
    sys.exit(1)

from schemas import SCHEMAS, SUPPORTED_CARD_TYPES
from mrz_validator import validate_mrz

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "minicpm-v4.6"
REQUEST_TIMEOUT_SEC = 180


def _check_server():
    try:
        requests.get("http://localhost:11434/api/version", timeout=5).raise_for_status()
    except requests.exceptions.ConnectionError:
        print("Can't reach Ollama at localhost:11434 — is it running? Start it with: ollama serve")
        sys.exit(1)


def _build_prompt(card_type: str) -> str:
    schema = SCHEMAS[card_type]
    fields = [k for k in schema if k != "card_type"]
    field_list = "\n".join(f"  - {f}" for f in fields)
    return f"""You are a KYC document parser. Extract fields from this {card_type.replace("_", " ")} image.

Fields to extract:
{field_list}

Rules:
- Return ONLY a valid JSON object with these exact field names as keys. No explanation, no markdown fences.
- Use null for any field that is not visible, not present, or illegible.
- Date fields: DD/MM/YYYY format.
- MRZ lines: copy EXACTLY including '<' fill characters. Each line must be 44 characters.
- Names: uppercase, as printed on the document.
- Do not invent or guess values — null is better than a wrong value."""


def _extract_fields(image_path: str, card_type: str) -> dict:
    prompt = _build_prompt(card_type)
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [img_b64],
            "format": "json",
            "stream": False,
        },
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    raw = resp.json()["response"].strip()
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

    _check_server()

    input_path = Path(input_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts)

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Labeling {len(images)} images as '{card_type}' via Ollama ({MODEL_NAME}, local, no rate limit)...")

    errors, mrz_flags = [], []
    with open(output_file, "w", encoding="utf-8") as out:
        for img_path in tqdm(images):
            img_name = img_path.name
            try:
                fields = _extract_fields(str(img_path), card_type)
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

            except json.JSONDecodeError as e:
                errors.append((img_name, f"model didn't return valid JSON: {e}"))
                print(f"\nError on {img_name}: model didn't return valid JSON: {e}")
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
    print("Reminder: this is zero-shot, unverified accuracy — every image still needs the normal "
          "Label Studio verification pass, same as the Gemini path.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
