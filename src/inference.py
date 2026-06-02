import torch
import json
import re
import time
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from mrz_validator import validate_mrz

import os

# Set path relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "donut-passport-final")
TASK_PROMPT = "<s_indian_passport>"


def load_model(model_path: str = MODEL_PATH):
    """Load the fine-tuned Donut model and processor."""
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


def backfill_from_mrz(passport: dict) -> dict:
    """Extract missing fields from MRZ if available."""
    # Line 1 (Names)
    m1 = passport.get("mrz_line1", "")
    if len(m1) >= 44:
        # Chars 5+ contain the name parts
        names_part = m1[5:].split("<<")
        if len(names_part) >= 1 and not passport.get("surname"):
            passport["surname"] = names_part[0].replace("<", " ").strip()
        if len(names_part) >= 2 and not passport.get("given_names"):
            passport["given_names"] = names_part[1].replace("<", " ").strip()

    # Line 2 (Meta)
    m2 = passport.get("mrz_line2", "")
    if len(m2) >= 44:
        # Passport Number
        if not passport.get("passport_number"):
            passport["passport_number"] = m2[0:9].replace("<", "").strip()
        
        # DOB (format YYMMDD -> DD/MM/YYYY)
        if not passport.get("dob") or passport.get("dob") == "":
            raw_dob = m2[13:19]
            if raw_dob.isdigit():
                yy, mm, dd = raw_dob[0:2], raw_dob[2:4], raw_dob[4:6]
                year_prefix = "19" if int(yy) > 25 else "20"
                passport["dob"] = f"{dd}/{mm}/{year_prefix}{yy}"
        
        # Sex
        if not passport.get("sex"):
            sex = m2[20]
            if sex in ("M", "F"):
                passport["sex"] = sex

    return passport


def clean_output(raw: dict) -> dict:
    """
    Clean up the token2json output:
    - Strip stray XML-style tokens (e.g. </s_country_code>) from field values.
    - Merge any top-level fields back into the indian_passport block.
    - Return a clean, flat passport dict.
    """
    TOKEN_PATTERN = re.compile(r"</?\w+>")

    def clean_value(v):
        if isinstance(v, str):
            return TOKEN_PATTERN.sub("", v).strip()
        if isinstance(v, dict):
            return {k: clean_value(val) for k, val in v.items()}
        return v

    # Get the main passport block
    passport = raw.get("indian_passport", {})
    if isinstance(passport, str):
        passport = {}

    passport = clean_value(passport)

    # Merge any stray top-level fields that leaked outside the passport block
    for key, value in raw.items():
        if key != "indian_passport" and key not in passport:
            passport[key] = clean_value(value)

    # Intelligent Backfill
    passport = backfill_from_mrz(passport)

    # Strictly enforce 44-character MRZ lines (pad if short, truncate if long)
    mrz1 = passport.get("mrz_line1")
    mrz2 = passport.get("mrz_line2")
    
    if mrz1:
        mrz1 = mrz1.strip()
        if len(mrz1) < 44:
            mrz1 = mrz1.ljust(44, "<")
        elif len(mrz1) > 44:
            mrz1 = mrz1[:44]
        passport["mrz_line1"] = mrz1

    if mrz2:
        mrz2 = mrz2.strip()
        if len(mrz2) < 44:
            mrz2 = mrz2.ljust(44, "<")
        elif len(mrz2) > 44:
            mrz2 = mrz2[:44]
        passport["mrz_line2"] = mrz2

    if mrz1 and mrz2:
        passport["mrz_validation"] = validate_mrz(mrz1, mrz2)
    else:
        passport["mrz_validation"] = {"valid": False, "reason": "One or both MRZ lines missing"}

    return {"indian_passport": passport}


def run_inference(image_path: str, model=None, processor=None, device=None) -> dict:
    """
    Run inference on a passport image.
    """
    print(f"\n--- Starting Inference ---")
    start_time = time.time()
    
    if model is None or processor is None:
        print("[1/5] Loading model and processor...")
        model, processor, device = load_model()
    else:
        print("[1/5] Using pre-loaded model weights.")

    print(f"[2/5] Opening and resizing image: {os.path.basename(image_path)}")
    image = Image.open(image_path).convert("RGB")
    
    print(f"[3/5] Pre-processing image (device: {device})...")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    print(f"[4/5] Running model generation (this is the heavy step)...")
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
    gen_time = time.time() - gen_start
    print(f"      Model generation finished in {gen_time:.2f}s")

    print(f"[5/5] Decoding and cleaning output...")
    raw_prediction = processor.batch_decode(outputs.sequences)[0]
    parsed = processor.token2json(raw_prediction)
    cleaned = clean_output(parsed)
    
    total_time = time.time() - start_time
    cleaned["execution_time_sec"] = round(total_time, 2)
    print(f"--- Inference Complete (Total: {total_time:.2f}s) ---\n")
    return cleaned


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_passport_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = run_inference(image_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))