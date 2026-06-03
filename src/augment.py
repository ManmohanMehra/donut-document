import albumentations as A
import numpy as np
import json, uuid, shutil
from PIL import Image
from pathlib import Path

# Simulate real-world passport photo conditions
# Includes degradation augmentations to handle bad quality images
augment = A.Compose([
    # --- Geometric ---
    A.Rotate(limit=8, p=0.8),
    A.Perspective(scale=(0.02, 0.06), p=0.6),

    # --- Colour & Light ---
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.4),
    A.RandomShadow(p=0.3),                                    # Simulate hand/lighting shadow

    # --- Noise & Blur ---
    A.GaussNoise(std_range=(0.1, 0.5), p=0.5),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),  # Phone ISO noise
    A.MotionBlur(blur_limit=5, p=0.4),
    A.Sharpen(alpha=(0.0, 0.5), lightness=(0.5, 1.0), p=0.3), # Over-sharpened phone processing

    # --- Compression & Resolution ---
    A.Downscale(scale_range=(0.4, 0.7), p=0.4),         # Simulate low-res camera (Albumentations 2.0 syntax)
    A.ImageCompression(quality_range=(40, 100), p=0.5),        # Aggressive WhatsApp compression

    # --- Occlusion ---
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(10, 20), hole_width_range=(20, 40), p=0.2), 

    # --- Glare / Overexposure ---
    A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.2),             # Glare sim
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
