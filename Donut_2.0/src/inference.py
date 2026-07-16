import torch
import json
import re
import time
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from mrz_validator import validate_mrz
from schemas import SCHEMAS, SUPPORTED_CARD_TYPES

import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "donut-multitype-final")

# Passport types that carry a TD3 MRZ (two 44-char lines)
_MRZ_TYPES = {ct for ct, fields in SCHEMAS.items() if "mrz_line1" in fields}


def load_model(model_path: str = MODEL_PATH):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model from: {model_path}  (device: {device})")
    processor = DonutProcessor.from_pretrained(model_path, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    return model, processor, device


def _backfill_from_mrz(doc: dict) -> dict:
    """Fill missing fields from MRZ lines (passports only)."""
    m1 = doc.get("mrz_line1", "")
    if len(m1) >= 44:
        names_part = m1[5:].split("<<")
        if len(names_part) >= 1 and not doc.get("surname"):
            doc["surname"] = names_part[0].replace("<", " ").strip()
        if len(names_part) >= 2 and not doc.get("given_names"):
            doc["given_names"] = names_part[1].replace("<", " ").strip()

    m2 = doc.get("mrz_line2", "")
    if len(m2) >= 44:
        if not doc.get("document_number"):
            doc["document_number"] = m2[0:9].replace("<", "").strip()
        if not doc.get("dob"):
            raw_dob = m2[13:19]
            if raw_dob.isdigit():
                yy, mm, dd = raw_dob[0:2], raw_dob[2:4], raw_dob[4:6]
                prefix = "19" if int(yy) > 25 else "20"
                doc["dob"] = f"{dd}/{mm}/{prefix}{yy}"
        if not doc.get("sex"):
            sex = m2[20]
            if sex in ("M", "F"):
                doc["sex"] = sex

    return doc


def _enforce_mrz_length(doc: dict) -> dict:
    """Pad or truncate MRZ lines to exactly 44 characters."""
    for key in ("mrz_line1", "mrz_line2"):
        val = doc.get(key)
        if val:
            val = val.strip()
            if len(val) < 44:
                val = val.ljust(44, "<")
            elif len(val) > 44:
                val = val[:44]
            doc[key] = val
    return doc


def clean_output(raw: dict, card_type: str) -> dict:
    """
    Generic post-processing for any card type:
    - Strip stray XML tokens from field values
    - Merge any leaked top-level fields back into the card block
    - MRZ backfill + validation for passport types
    """
    TOKEN_PATTERN = re.compile(r"</?\w+>")

    def clean_value(v):
        if isinstance(v, str):
            return TOKEN_PATTERN.sub("", v).strip()
        if isinstance(v, dict):
            return {k: clean_value(val) for k, val in v.items()}
        return v

    doc = raw.get(card_type, {})
    if isinstance(doc, str):
        doc = {}
    doc = clean_value(doc)

    # Merge any stray top-level fields that leaked outside the card block
    for key, value in raw.items():
        if key != card_type and key not in doc:
            doc[key] = clean_value(value)

    if card_type in _MRZ_TYPES:
        doc = _backfill_from_mrz(doc)
        doc = _enforce_mrz_length(doc)
        mrz1 = doc.get("mrz_line1")
        mrz2 = doc.get("mrz_line2")
        if mrz1 and mrz2:
            doc["mrz_validation"] = validate_mrz(mrz1, mrz2)
        else:
            doc["mrz_validation"] = {"valid": False, "reason": "One or both MRZ lines missing"}

    return {card_type: doc}


def run_inference(
    image_path: str,
    card_type: str = "indian_passport",
    model=None,
    processor=None,
    device=None,
) -> dict:
    """
    Run inference on a document image.

    Args:
        image_path: path to the image file
        card_type:  one of SUPPORTED_CARD_TYPES; controls the decoder task prompt
    """
    if card_type not in SUPPORTED_CARD_TYPES:
        return {
            "status": "unsupported",
            "card_type": card_type,
            "message": (
                f"Card type '{card_type}' is not supported in this model version. "
                "Route to GPT-4V fallback or human review queue."
            ),
            "supported_types": SUPPORTED_CARD_TYPES,
        }

    print(f"\n--- Starting Inference [{card_type}] ---")
    start_time = time.time()

    if model is None or processor is None:
        print("[1/5] Loading model and processor...")
        model, processor, device = load_model()
    else:
        print("[1/5] Using pre-loaded model.")

    print(f"[2/5] Opening image: {os.path.basename(image_path)}")
    image = Image.open(image_path).convert("RGB")

    print(f"[3/5] Pre-processing (device: {device})...")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    task_prompt = f"<s_{card_type}>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    print("[4/5] Running model generation...")
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    print(f"      Generation done in {time.time() - gen_start:.2f}s")

    print("[5/5] Decoding and cleaning output...")
    raw_prediction = processor.batch_decode(outputs.sequences)[0]
    parsed = processor.token2json(raw_prediction)
    cleaned = clean_output(parsed, card_type)

    total_time = time.time() - start_time
    cleaned["execution_time_sec"] = round(total_time, 2)
    print(f"--- Inference Complete ({total_time:.2f}s) ---\n")
    return cleaned


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [card_type]")
        print(f"Supported card types: {SUPPORTED_CARD_TYPES}")
        sys.exit(1)

    image_path = sys.argv[1]
    card_type  = sys.argv[2] if len(sys.argv) > 2 else "indian_passport"
    result = run_inference(image_path, card_type=card_type)
    print(json.dumps(result, indent=2, ensure_ascii=False))
