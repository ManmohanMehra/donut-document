"""
pre_annotate_minicpm.py — Local pre-annotation using MiniCPM-V-4.6, as a free
alternative to vision_label.py's Gemini API calls.

Why this exists: at your target volume (800 images now, 10-20K later),
Gemini's free tier doesn't hold up — it deprioritizes free-tier traffic
under load (the 503 "high demand" errors you hit) and has hard daily
request caps regardless. The paid tier is genuinely cheap for this workload
(~$3-12 for 20K images via Batch API), but you chose to stay free. MiniCPM-V
run locally has no API cost and no rate limit at all — throughput is capped
only by your own GPU, and your earlier R&D chat already flagged this exact
model for exactly this role ("use it as a pre-annotation tool... zero-shot,
~75% accurate").

Output format is identical to vision_label.py's (same metadata_suggested.jsonl
shape, same _review_flag on MRZ checksum failure), so everything downstream
— Label Studio import, convert_labelstudio.py — needs zero changes regardless
of which backend produced the suggestions.

Hardware: ~4-8GB VRAM. Will run on CPU but slowly — not practical for
hundreds of images. Apple Silicon (MPS) support is not confirmed for this
model as of writing; if you're on a Mac and generation errors out or
silently produces garbage, that's likely why. Meant to run on your NTT A40
server, not locally on the Mac driving pre_annotate_all.py today — copy
data/passport_data/ and this repo's src/ over there.

Setup (on the GPU machine):
    pip install transformers accelerate pillow torch
    # first run downloads ~3GB of weights from Hugging Face to ~/.cache

Usage — same CLI shape as vision_label.py:
    python src/pre_annotate_minicpm.py data/ZWE  data/ZWE/metadata_suggested.jsonl  zwe_passport

NOTE ON UNVERIFIED PARTS: the exact processor.apply_chat_template() kwargs
below (downsample_mode, max_slice_nums) come from the model card, not from a
run against real hardware I have access to — I could not execute this
end-to-end myself. If loading/generation errors on first run, check
https://huggingface.co/openbmb/MiniCPM-V-4.6 for the current calling
convention and adjust _generate() accordingly; the rest of this script
(prompt building, JSON parsing, MRZ flagging, resumability) doesn't depend
on those specific kwargs and should still be correct.
"""
import json
import sys
from pathlib import Path
from tqdm import tqdm

from schemas import SCHEMAS, SUPPORTED_CARD_TYPES
from mrz_validator import validate_mrz

try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
except ImportError:
    print("Required packages not found. Run:")
    print("  pip install transformers accelerate pillow torch")
    sys.exit(1)

MODEL_ID = "openbmb/MiniCPM-V-4.6"

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    print(f"Loading {MODEL_ID} (first run downloads ~3GB)...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    device = next(_model.parameters()).device
    print(f"Model loaded on: {device}")
    return _model, _processor


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


def _generate(model, processor, image: "Image.Image", prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _extract_fields(model, processor, image_path: str, card_type: str) -> dict:
    prompt = _build_prompt(card_type)
    img = Image.open(image_path).convert("RGB")
    raw = _generate(model, processor, img, prompt)
    return _parse_json(raw)


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

    input_path = Path(input_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts)

    if not images:
        print(f"No images found in {input_dir}")
        return

    model, processor = _load_model()
    print(f"Labeling {len(images)} images as '{card_type}' — no rate limit, speed depends on your GPU.")

    errors, mrz_flags = [], []
    with open(output_file, "w", encoding="utf-8") as out:
        for img_path in tqdm(images):
            img_name = img_path.name
            try:
                fields = _extract_fields(model, processor, str(img_path), card_type)
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
    print("Reminder: this is zero-shot, ~75% accurate per your own earlier estimate — "
          "every image still needs the normal Label Studio verification pass, same as the Gemini path.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
