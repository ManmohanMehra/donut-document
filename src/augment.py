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
    augmentations_per_image: int = 40
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
        try:
            img = Image.open(src_img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Skipping {record['file_name']}, image not found.")
            continue
            
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

    print(f"Done. {len(originals)} originals \u2192 {len(records)} total samples")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    augment_dataset(
        real_dir="data/real",
        output_dir="data/augmented",
        augmentations_per_image=40    # 19 \u00d7 40 = 760 + 19 originals = 779 total
    )
