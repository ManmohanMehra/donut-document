# Passport & ID Card Parser — Project Reference
### Donut Fine-Tuning | Real Data + Augmentation | FastAPI Deployment

> **Personal initiative project.** Goal: replace GPT-4 Vision (~$10K/month) with a
> fine-tuned, self-hosted Donut model that accepts a passport/ID image and returns
> a structured JSON output. Built to learn MLOps end-to-end.

---

## Table of Contents

1. [Why Donut](#1-why-donut)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Environment Setup](#4-environment-setup)
5. [Step 1 — Define Schemas](#5-step-1--define-schemas)
6. [Step 2 — Annotate Real Samples (Label Studio)](#6-step-2--annotate-real-samples-label-studio)
7. [Step 3 — Augmentation Pipeline](#7-step-3--augmentation-pipeline)
8. [Step 4 — Register Custom Tokens](#8-step-4--register-custom-tokens)
9. [Step 5 — Dataset Class](#9-step-5--dataset-class)
10. [Step 6 — Training](#10-step-6--training)
11. [Step 7 — Inference + MRZ Validation](#11-step-7--inference--mrz-validation)
12. [Step 8 — FastAPI Server](#12-step-8--fastapi-server)
13. [Step 9 — Docker Deployment](#13-step-9--docker-deployment)
14. [Step 10 — Optimisation (Post-MVP)](#14-step-10--optimisation-post-mvp)
15. [Execution Order Cheatsheet](#15-execution-order-cheatsheet)
16. [Expected Performance](#16-expected-performance)
17. [Troubleshooting](#17-troubleshooting)
18. [Roadmap](#18-roadmap)

---

## 1. Why Donut

| | GPT-4 Vision | PaddleOCR + Parser | **Donut (This Project)** |
|---|---|---|---|
| Cost | ~$10K/month | Near zero | Near zero |
| Deployment | API only | Docker nightmare | Single container |
| Reliability | Hallucinations | Layout-dependent | Deterministic |
| Control | None | Partial | Full |
| Latency | 2–5s (network) | 300–600ms | 80–800ms |
| Learning value | None | Medium | High |

**How Donut works — one forward pass, no OCR:**

```
Image
  │
  ▼
SwinTransformer Encoder        ← Breaks image into visual patches
  │
  ▼ (cross-attention)
BartDecoder                    ← Autoregressively generates text
  │
  ▼
<s_indian_passport>
  <s_surname>SINGH</s_surname>
  <s_passport_number>A1234567</s_passport_number>
  ...
</s_indian_passport>
  │
  ▼
{"surname": "SINGH", "passport_number": "A1234567", ...}
```

No OCR stage. No layout detection. No post-processing pipeline.
One model, one Docker container, structured JSON out.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   REQUEST FLOW                      │
│                                                     │
│  Client ──POST /parse──► FastAPI                    │
│                              │                      │
│                         card_type?                  │
│                        /         \                  │
│                  known        unknown               │
│                    │               │                │
│                    │         Classifier             │
│                    │         (ResNet/CLIP)          │
│                    │               │                │
│                    └──────┬────────┘                │
│                           │                         │
│                      DonutParser                    │
│                     (fine-tuned)                    │
│                           │                         │
│                      JSON output                    │
│                           │                         │
│                    MRZ Checksum?                    │
│                   /            \                    │
│              PASS               FAIL                │
│           return JSON       flag + return           │
└─────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
donut-passport-parser/
│
├── data/
│   ├── real/
│   │   ├── images/                  # Your 50 passport photos
│   │   └── metadata.jsonl           # Annotated ground truth
│   └── augmented/
│       ├── images/                  # Augmented versions (750–1000 images)
│       └── metadata.jsonl           # Copied ground truth per augmented image
│
├── src/
│   ├── schemas.py                   # JSON field definitions per card type
│   ├── add_tokens.py                # Register custom tokens to Donut vocab
│   ├── augment.py                   # Real image augmentation pipeline
│   ├── dataset.py                   # PyTorch Dataset class
│   ├── train.py                     # Fine-tuning script
│   └── inference.py                 # IDCardParser class + MRZ validation
│
├── serve/
│   ├── app.py                       # FastAPI application
│   └── Dockerfile
│
├── checkpoints/
│   ├── donut-passport-processor/    # Saved processor with custom tokens
│   └── donut-passport-final/        # Final fine-tuned model weights
│
├── requirements.txt
└── README.md
```

---

## 4. Environment Setup

### 4a. Python environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 4b. requirements.txt

```txt
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.37.0
datasets>=2.16.0
Pillow>=10.0.0
albumentations>=1.3.0
faker>=20.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
wandb>=0.16.0
sentencepiece>=0.1.99
```

### 4c. Hardware recommendation

| Environment | Use For |
|---|---|
| MacBook Air M5 (local) | Annotation, code, quick inference tests |
| GCP CUDA instance (T4/A100) | Training only |
| Kaggle/Colab | Prototyping, sanity checks |

> **Training on CPU is not recommended.** Use your GCP credits for the training run.
> A T4 instance handles this dataset comfortably in 1–2 hours.

### 4d. WandB setup (for training monitoring)

```bash
pip install wandb
wandb login     # Paste your API key from wandb.ai
```

---

## 5. Step 1 — Define Schemas

Define the exact JSON fields you want to extract per card type.
**Do this before annotation** — it determines what you annotate.

```python
# src/schemas.py

SCHEMAS = {

    "indian_passport": {
        "card_type": "indian_passport",
        "surname": None,
        "given_names": None,
        "passport_number": None,
        "nationality": None,
        "dob": None,               # DD/MM/YYYY
        "sex": None,               # M / F
        "place_of_birth": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "mrz_line1": None,         # Raw MRZ string — used for checksum validation
        "mrz_line2": None,
    },

    "foreign_passport": {
        "card_type": "foreign_passport",
        "issuing_country": None,   # 3-letter ISO code e.g. GBR, USA, DEU
        "surname": None,
        "given_names": None,
        "passport_number": None,
        "nationality": None,
        "dob": None,
        "sex": None,
        "date_of_expiry": None,
        "mrz_line1": None,
        "mrz_line2": None,
    },

    # Add future card types here following the same pattern
    # "aadhaar": { ... }
    # "pan": { ... }
}
```

> **Design principle:** Always include `mrz_line1` and `mrz_line2` for any passport
> type. The MRZ is machine-readable and gives you a free accuracy check via checksum.

---

## 6. Step 2 — Annotate Real Samples (Label Studio)

You have 50 real Indian passport images. These need ground truth JSON before training.

### 6a. Install and launch Label Studio

```bash
pip install label-studio
label-studio start
# Opens at http://localhost:8080
```

### 6b. Label Studio project config

Create a new project → use this labeling interface XML:

```xml
<View>
  <Image name="image" value="$image"/>
  <TextArea name="surname"         toName="image" label="Surname"/>
  <TextArea name="given_names"     toName="image" label="Given Names"/>
  <TextArea name="passport_number" toName="image" label="Passport Number"/>
  <TextArea name="nationality"     toName="image" label="Nationality"/>
  <TextArea name="dob"             toName="image" label="Date of Birth (DD/MM/YYYY)"/>
  <TextArea name="sex"             toName="image" label="Sex (M/F)"/>
  <TextArea name="place_of_birth"  toName="image" label="Place of Birth"/>
  <TextArea name="date_of_issue"   toName="image" label="Date of Issue"/>
  <TextArea name="date_of_expiry"  toName="image" label="Date of Expiry"/>
  <TextArea name="mrz_line1"       toName="image" label="MRZ Line 1"/>
  <TextArea name="mrz_line2"       toName="image" label="MRZ Line 2"/>
</View>
```

### 6c. Export and convert annotations

After annotating all 50, export as **JSON** from Label Studio, then convert:

```python
# src/convert_labelstudio.py
import json
from pathlib import Path

def convert_ls_export(ls_export_path: str, output_jsonl: str):
    with open(ls_export_path) as f:
        annotations = json.load(f)

    records = []
    for item in annotations:
        file_name = Path(item["data"]["image"]).name
        result = item["annotations"][0]["result"]

        ground_truth = {"card_type": "indian_passport"}
        for field in result:
            key = field["from_name"]
            value = field["value"]["text"][0] if field["value"]["text"] else ""
            ground_truth[key] = value.strip()

        records.append({"file_name": file_name, "ground_truth": ground_truth})

    with open(output_jsonl, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Converted {len(records)} annotations → {output_jsonl}")

if __name__ == "__main__":
    convert_ls_export(
        "data/real/label_studio_export.json",
        "data/real/metadata.jsonl"
    )
```

> **Time estimate:** ~2–3 hours to annotate 50 passports carefully.
> Type MRZ lines character-by-character — accuracy here is critical.

---

## 7. Step 3 — Augmentation Pipeline

Transform 50 real annotated images into 750–1000 training samples.
No fake text overlays. The ground truth JSON stays identical per original image —
only the visual changes.

```python
# src/augment.py
import albumentations as A
import numpy as np
import json, uuid, shutil
from PIL import Image
from pathlib import Path

# Simulate real-world passport photo conditions
augment = A.Compose([
    A.Rotate(limit=8, p=0.8),
    A.Perspective(scale=(0.02, 0.06), p=0.6),
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.7
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ImageCompression(quality_lower=65, p=0.4),
    A.CoarseDropout(
        max_holes=3,
        max_height=20,
        max_width=40,
        p=0.2                      # Simulate thumb/finger covering part of card
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        p=0.4                      # Colour temperature variation from different phones
    ),
])

def augment_dataset(
    real_dir: str,
    output_dir: str,
    augmentations_per_image: int = 15
):
    real_path = Path(real_dir)
    out_path = Path(output_dir)
    (out_path / "images").mkdir(parents=True, exist_ok=True)

    records = []

    # Load original annotations
    with open(real_path / "metadata.jsonl") as f:
        originals = [json.loads(line) for line in f]

    for record in originals:
        src_img_path = real_path / "images" / record["file_name"]
        img = Image.open(src_img_path).convert("RGB")
        img_np = np.array(img)

        # Keep one clean copy of the original
        clean_filename = f"clean_{record['file_name']}"
        shutil.copy(src_img_path, out_path / "images" / clean_filename)
        records.append({
            "file_name": clean_filename,
            "ground_truth": record["ground_truth"]
        })

        # Generate augmented versions
        for _ in range(augmentations_per_image):
            augmented = augment(image=img_np)["image"]
            aug_img = Image.fromarray(augmented)

            filename = f"aug_{uuid.uuid4().hex[:10]}.jpg"
            aug_img.save(out_path / "images" / filename, quality=92)

            records.append({
                "file_name": filename,
                "ground_truth": record["ground_truth"]   # Same GT as original
            })

    # Save combined metadata
    with open(out_path / "metadata.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Done. {len(originals)} originals → {len(records)} total samples")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    augment_dataset(
        real_dir="data/real",
        output_dir="data/augmented",
        augmentations_per_image=15    # 50 × 15 = 750 + 50 originals = 800 total
    )
```

> **What augmentations simulate:**
> - `Rotate` → hand-held camera tilt
> - `Perspective` → card not flat on table
> - `BrightnessContrast` → indoor vs outdoor lighting
> - `GaussNoise` → phone camera sensor noise
> - `MotionBlur` → slight movement during capture
> - `ImageCompression` → WhatsApp/JPEG re-encoding
> - `CoarseDropout` → thumb partially covering card
> - `HueSaturationValue` → different phone colour profiles

---

## 8. Step 4 — Register Custom Tokens

Donut's vocabulary must know your structural tokens before training.
**Run this once. Re-run only when adding new card types.**

```python
# src/add_tokens.py
from transformers import DonutProcessor

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

new_tokens = [
    # Indian Passport
    "<s_indian_passport>", "</s_indian_passport>",
    "<s_surname>", "</s_surname>",
    "<s_given_names>", "</s_given_names>",
    "<s_passport_number>", "</s_passport_number>",
    "<s_nationality>", "</s_nationality>",
    "<s_dob>", "</s_dob>",
    "<s_sex>", "</s_sex>",
    "<s_place_of_birth>", "</s_place_of_birth>",
    "<s_date_of_issue>", "</s_date_of_issue>",
    "<s_date_of_expiry>", "</s_date_of_expiry>",
    "<s_mrz_line1>", "</s_mrz_line1>",
    "<s_mrz_line2>", "</s_mrz_line2>",

    # Foreign Passport
    "<s_foreign_passport>", "</s_foreign_passport>",
    "<s_issuing_country>", "</s_issuing_country>",

    # Add new card type tokens here as you expand
]

processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
processor.save_pretrained("checkpoints/donut-passport-processor")

print(f"New vocab size: {len(processor.tokenizer)}")
# Should be: original (~57,522) + number of new tokens you added
```

---

## 9. Step 5 — Dataset Class

```python
# src/dataset.py
from torch.utils.data import Dataset
from transformers import DonutProcessor
from PIL import Image
import json, random
from pathlib import Path

class PassportDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        processor: DonutProcessor,
        split: str = "train",
        max_length: int = 512,
        val_split: float = 0.1
    ):
        self.processor = processor
        self.max_length = max_length
        self.data_dir = Path(data_dir)

        with open(self.data_dir / "metadata.jsonl") as f:
            records = [json.loads(line) for line in f]

        # Reproducible split
        random.seed(42)
        random.shuffle(records)
        split_idx = int(len(records) * (1 - val_split))
        self.records = records[:split_idx] if split == "train" else records[split_idx:]

        print(f"[{split}] {len(self.records)} samples loaded")

    def __len__(self):
        return len(self.records)

    def _gt_to_token_sequence(self, gt: dict) -> str:
        """
        {"card_type": "indian_passport", "surname": "SINGH", ...}
        →
        <s_indian_passport><s_surname>SINGH</s_surname>...</s_indian_passport>
        """
        gt = gt.copy()
        card_type = gt.pop("card_type")
        seq = f"<s_{card_type}>"
        for key, value in gt.items():
            seq += f"<s_{key}>{value or ''}</s_{key}>"
        seq += f"</s_{card_type}>"
        return seq

    def __getitem__(self, idx):
        record = self.records[idx]

        # Load and process image
        img = Image.open(
            self.data_dir / "images" / record["file_name"]
        ).convert("RGB")
        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze()

        # Build and tokenize target sequence
        target_seq = self._gt_to_token_sequence(record["ground_truth"])
        labels = self.processor.tokenizer(
            target_seq,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Mask padding from loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
```

> **Debug tip:** Print a few `target_seq` strings before training to verify
> they look like `<s_indian_passport><s_surname>SINGH</s_surname>...`.
> Wrong token sequences are the #1 cause of poor training.

---

## 10. Step 6 — Training

```python
# src/train.py
import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataset import PassportDataset

# ── Load processor and model ────────────────────────────────────────────────
processor = DonutProcessor.from_pretrained("checkpoints/donut-passport-processor")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Expand decoder embeddings to match new vocab size
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Decoder config — set start token to your primary card type
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
    ["<s_indian_passport>"]
)[0]
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset = PassportDataset("data/augmented", processor, split="train")
val_dataset   = PassportDataset("data/augmented", processor, split="val")

# ── Training arguments ───────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints/donut-passport-finetuned",

    # Epochs: Donut converges fast on small focused datasets
    num_train_epochs=30,

    # Batch size: reduce to 2 if you hit OOM on T4
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,

    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    predict_with_generate=True,

    # Precision: fp16 for T4, bf16 for A100
    fp16=True,
    # bf16=True,

    logging_steps=25,
    report_to="wandb",
    run_name="donut-passport-v1",

    dataloader_num_workers=4,
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model("checkpoints/donut-passport-final")
processor.save_pretrained("checkpoints/donut-passport-final")
print("Training complete. Model saved to checkpoints/donut-passport-final")
```

### Reading the loss curves (WandB)

| What you see | What it means | Action |
|---|---|---|
| Train loss falling, val loss following | Healthy training | Let it run |
| Val loss stops improving after epoch 10 | Early convergence | Stop, use best checkpoint |
| Train loss falls, val loss rises | Overfitting | More augmentation, reduce epochs |
| Both losses plateau high | Bad token sequences | Debug `_gt_to_token_sequence` output |

---

## 11. Step 7 — Inference + MRZ Validation

### 11a. MRZ checksum validator

Indian (and all ICAO) passports embed checksums in the MRZ.
Use this as a free confidence score — if it fails, flag the result.

```python
# src/mrz_validator.py

MRZ_WEIGHTS = [7, 3, 1]
MRZ_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"

def mrz_check_digit(s: str) -> int:
    """Compute ICAO MRZ check digit for a string."""
    total = 0
    for i, ch in enumerate(s):
        if ch not in MRZ_CHARS:
            return -1   # Invalid character
        val = int(ch) if ch.isdigit() else (0 if ch == "<" else ord(ch) - 55)
        total += val * MRZ_WEIGHTS[i % 3]
    return total % 10

def validate_mrz(line1: str, line2: str) -> dict:
    """
    Validate key check digits in an ICAO TD3 (passport) MRZ.
    Returns a dict of field → pass/fail.
    """
    if len(line1) != 44 or len(line2) != 44:
        return {"valid": False, "reason": "MRZ lines must be 44 characters each"}

    checks = {}

    # Passport number (chars 1–9, check digit at 10)
    checks["passport_number"] = mrz_check_digit(line2[0:9]) == int(line2[9])

    # Date of birth (chars 14–19, check at 20)
    checks["dob"] = mrz_check_digit(line2[13:19]) == int(line2[19])

    # Date of expiry (chars 22–27, check at 28)
    checks["expiry"] = mrz_check_digit(line2[21:27]) == int(line2[27])

    # Overall composite check (chars 1–10, 14–20, 22–43 of line2)
    composite = line2[0:10] + line2[13:20] + line2[21:43]
    checks["composite"] = mrz_check_digit(composite) == int(line2[43])

    checks["valid"] = all(checks.values())
    return checks
```

### 11b. IDCardParser class

```python
# src/inference.py
import torch, re, json
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from mrz_validator import validate_mrz

class IDCardParser:
    def __init__(self, model_path: str):
        self.processor = DonutProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def parse(self, image_path: str, card_type: str = "indian_passport") -> dict:
        img = Image.open(image_path).convert("RGB")

        # Prime the decoder with the card type start token
        decoder_input = f"<s_{card_type}>"
        decoder_input_ids = self.processor.tokenizer(
            decoder_input,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,         # Greedy (fast). Use num_beams=4 for +accuracy
            )

        sequence = self.processor.batch_decode(outputs)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")

        result = self._decode_sequence(sequence, card_type)

        # MRZ validation for passport types
        if card_type in ("indian_passport", "foreign_passport"):
            mrz1 = result.get("mrz_line1", "")
            mrz2 = result.get("mrz_line2", "")
            if mrz1 and mrz2:
                result["mrz_validation"] = validate_mrz(mrz1, mrz2)
            else:
                result["mrz_validation"] = {"valid": False, "reason": "MRZ not extracted"}

        return result

    def _decode_sequence(self, sequence: str, card_type: str) -> dict:
        result = {"card_type": card_type}
        pattern = r"<s_(\w+)>(.*?)</s_\1>"
        for field, value in re.findall(pattern, sequence, re.DOTALL):
            if field != card_type:
                result[field] = value.strip()
        return result


if __name__ == "__main__":
    parser = IDCardParser("checkpoints/donut-passport-final")
    out = parser.parse("test_passport.jpg", card_type="indian_passport")
    print(json.dumps(out, indent=2))
```

**Example output:**
```json
{
  "card_type": "indian_passport",
  "surname": "SINGH",
  "given_names": "MANMOHAN",
  "passport_number": "A1234567",
  "nationality": "INDIAN",
  "dob": "01/01/1995",
  "sex": "M",
  "place_of_birth": "AMRITSAR",
  "date_of_issue": "01/01/2020",
  "date_of_expiry": "31/12/2030",
  "mrz_line1": "P<INDSINGH<<MANMOHAN<<<<<<<<<<<<<<<<<<<<<<<<",
  "mrz_line2": "A12345671IND9501011M3012319<<<<<<<<<<<<<<<6",
  "mrz_validation": {
    "passport_number": true,
    "dob": true,
    "expiry": true,
    "composite": true,
    "valid": true
  }
}
```

---

## 12. Step 8 — FastAPI Server

```python
# serve/app.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import sys
sys.path.append("/app/src")
from inference import IDCardParser

app = FastAPI(
    title="Passport & ID Card Parser",
    description="Donut-based document understanding. One image in, structured JSON out.",
    version="1.0.0"
)

# Load model once at startup
parser = IDCardParser("/app/checkpoints/donut-passport-final")

SUPPORTED_CARD_TYPES = ["indian_passport", "foreign_passport"]
SUPPORTED_MIME = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
def health():
    return {"status": "ok", "supported_card_types": SUPPORTED_CARD_TYPES}


@app.post("/parse")
async def parse_card(
    file: UploadFile = File(...),
    card_type: str = Query(default="indian_passport", enum=SUPPORTED_CARD_TYPES)
):
    if file.content_type not in SUPPORTED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP."
        )

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Save to temp path (IDCardParser reads from disk)
    tmp_path = "/tmp/input_card.jpg"
    img.save(tmp_path, format="JPEG")

    result = parser.parse(tmp_path, card_type=card_type)
    return JSONResponse(content=result)
```

**Test with curl:**
```bash
curl -X POST "http://localhost:8000/parse?card_type=indian_passport" \
  -F "file=@my_passport.jpg"
```

---

## 13. Step 9 — Docker Deployment

```dockerfile
# serve/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY serve/ ./serve/

# Model checkpoint — bake in or mount as volume
# Option A: Bake in (larger image, simpler)
COPY checkpoints/donut-passport-final/ ./checkpoints/donut-passport-final/

# Option B (preferred for dev): Mount at runtime
# docker run -v /local/checkpoints:/app/checkpoints ...

EXPOSE 8000

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**Build and run:**
```bash
# Build
docker build -t donut-passport-parser -f serve/Dockerfile .

# Run (with baked-in model)
docker run -p 8000:8000 donut-passport-parser

# Run (with volume-mounted model — faster iteration during dev)
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  donut-passport-parser

# Test
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/parse?card_type=indian_passport" \
  -F "file=@test_passport.jpg"
```

> **Why `--workers 1`?** The Donut model is not thread-safe for concurrent
> inference. For production scale, run multiple container replicas behind a
> load balancer instead of multiple uvicorn workers.

---

## 14. Step 10 — Optimisation (Post-MVP)

Do these only **after** you have a working, accurate baseline.

### 14a. INT8 Quantization (cuts model size ~4×, CPU latency ~2×)

```bash
pip install optimum[onnxruntime]
```

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import DonutProcessor

processor = DonutProcessor.from_pretrained("checkpoints/donut-passport-final")
model = ORTModelForSeq2SeqLM.from_pretrained(
    "checkpoints/donut-passport-final",
    export=True,
    provider="CPUExecutionProvider"
)
model.save_pretrained("checkpoints/donut-passport-onnx")
```

### 14b. Card type auto-detection

When the caller doesn't know the card type, add a lightweight classifier:

```python
# Option 1: Zero-shot with CLIP (no training needed, ~1s overhead)
from transformers import CLIPProcessor, CLIPModel
import torch

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def detect_card_type(image_path: str) -> str:
    img = Image.open(image_path)
    labels = ["Indian passport", "foreign passport", "Aadhaar card", "PAN card"]
    inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image
    predicted = labels[logits.argmax().item()]
    # Map to your schema keys
    mapping = {
        "Indian passport": "indian_passport",
        "foreign passport": "foreign_passport",
    }
    return mapping.get(predicted, "indian_passport")
```

### 14c. Adding new card types later

1. Add schema to `schemas.py`
2. Add tokens to `add_tokens.py` → re-run it
3. Collect and annotate samples
4. Run `augment.py` on new samples
5. **Merge** new augmented data with existing `metadata.jsonl`
6. Retrain from your last checkpoint (not from scratch)

---

## 15. Execution Order Cheatsheet

```
PHASE 1 — DATA (do locally on MacBook)
────────────────────────────────────────────────────────
□ 1. pip install -r requirements.txt

□ 2. Define schemas
     edit src/schemas.py

□ 3. Annotate 50 real passport images
     label-studio start
     → annotate → export JSON
     python src/convert_labelstudio.py

□ 4. Run augmentation
     python src/augment.py
     # Output: data/augmented/ with ~800 samples

□ 5. Register custom tokens
     python src/add_tokens.py
     # Output: checkpoints/donut-passport-processor/

□ 6. Debug dataset (sanity check)
     python -c "
     from transformers import DonutProcessor
     from src.dataset import PassportDataset
     p = DonutProcessor.from_pretrained('checkpoints/donut-passport-processor')
     ds = PassportDataset('data/augmented', p, split='train')
     sample = ds[0]
     print('pixel_values shape:', sample['pixel_values'].shape)
     print('labels shape:', sample['labels'].shape)
     "


PHASE 2 — TRAINING (run on GCP CUDA instance)
────────────────────────────────────────────────────────
□ 7. Upload data + code to GCP
     rsync -avz data/ user@gcp-ip:~/donut-project/data/
     rsync -avz src/  user@gcp-ip:~/donut-project/src/
     rsync -avz checkpoints/donut-passport-processor/ \
           user@gcp-ip:~/donut-project/checkpoints/donut-passport-processor/

□ 8. Train
     python src/train.py
     # Monitor: https://wandb.ai

□ 9. Download final checkpoint
     rsync -avz user@gcp-ip:~/donut-project/checkpoints/donut-passport-final/ \
           checkpoints/donut-passport-final/


PHASE 3 — DEPLOY (local or any server)
────────────────────────────────────────────────────────
□ 10. Sanity check inference
      python src/inference.py
      # Should print valid JSON for a test passport image

□ 11. Build and run Docker
      docker build -t donut-passport-parser -f serve/Dockerfile .
      docker run -p 8000:8000 donut-passport-parser

□ 12. Smoke test
      curl http://localhost:8000/health
      curl -X POST "http://localhost:8000/parse?card_type=indian_passport" \
        -F "file=@test_passport.jpg"
```

---

## 16. Expected Performance

| Metric | Realistic Target |
|---|---|
| Field extraction accuracy (clean scans) | 90–95% |
| Field extraction accuracy (phone photos) | 80–88% |
| MRZ extraction accuracy | 95%+ (high contrast, fixed format) |
| Inference time — CPU | 400–800ms |
| Inference time — GPU (T4) | 80–150ms |
| Model size (full) | ~750MB |
| Model size (INT8 ONNX) | ~190MB |
| Training data needed | 300–500 images minimum |

---

## 17. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `KeyError` on token during training | Custom tokens not registered | Re-run `add_tokens.py` before training |
| Very high validation loss from epoch 1 | Wrong token sequence format | Print and inspect `_gt_to_token_sequence` output |
| Model outputs garbled text | `decoder_start_token_id` mismatch | Set it to your primary card type's start token |
| OOM on GCP T4 | Batch size too large | Reduce `per_device_train_batch_size` to 2 |
| MRZ validation always fails | MRZ annotated with spaces | Strip spaces from MRZ lines before saving GT |
| Docker container crashes on startup | Model path wrong | Check volume mount path or `COPY` destination |
| Inference output missing fields | `max_length` too short | Increase `max_length` in both dataset and inference |

---

## 18. Roadmap

```
v1.0 — MVP
  ✅ Indian passport, end-to-end
  ✅ MRZ validation
  ✅ FastAPI + Docker

v1.1 — Multi-card
  ○ Foreign passports (use existing MRZ infrastructure)
  ○ CLIP-based card type auto-detection

v1.2 — Robustness
  ○ Confidence scoring (beam search + probability)
  ○ Human review queue for low-confidence results
  ○ Evaluation script: field-level F1 on held-out test set

v2.0 — Optimisation
  ○ INT8 ONNX export
  ○ Aadhaar and PAN card support
  ○ Horizontal scaling (multiple container replicas)
```

---

*Built as a personal initiative to replace GPT-4 Vision with a self-hosted,
cost-effective, and controllable document understanding pipeline.*
*Stack: Donut · PyTorch · HuggingFace Transformers · FastAPI · Docker*
