import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
from inference import load_model, run_inference
from schemas import SUPPORTED_CARD_TYPES


def find_images(directory):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
        if Path(file).suffix.lower() in extensions
    ]


def main(input_path: str, output_dir: str = "batch_results", card_type: str = "indian_passport"):
    if card_type not in SUPPORTED_CARD_TYPES:
        print(f"Unsupported card_type '{card_type}'. Supported: {SUPPORTED_CARD_TYPES}")
        return

    os.makedirs(output_dir, exist_ok=True)
    images = find_images(input_path)
    if not images:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(images)} images. Card type: {card_type}. Loading model...")
    model, processor, device = load_model()

    results = []
    for img_path in tqdm(images):
        try:
            result = run_inference(img_path, card_type=card_type, model=model, processor=processor, device=device)
            results.append({
                "filename": os.path.basename(img_path),
                "relative_path": os.path.relpath(img_path, input_path),
                "prediction": result.get(card_type, {}),
                "execution_time_sec": result.get("execution_time_sec"),
            })
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    # JSON output
    json_output = os.path.join(output_dir, "results.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV output
    csv_output = os.path.join(output_dir, "results.csv")
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r["prediction"].keys())
        fieldnames = ["filename", "relative_path", "execution_time_sec"] + sorted(all_keys)

        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "filename": r["filename"],
                    "relative_path": r["relative_path"],
                    "execution_time_sec": r["execution_time_sec"],
                }
                for k, v in r["prediction"].items():
                    row[k] = str(v) if isinstance(v, (dict, list)) else v
                writer.writerow(row)

    print(f"\nDone! Processed {len(images)} images.")
    print(f"JSON: {json_output}")
    print(f"CSV:  {csv_output}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/batch_inference.py <input_folder> [output_folder] [card_type]")
        print(f"Supported card types: {SUPPORTED_CARD_TYPES}")
    else:
        in_path   = sys.argv[1]
        out_path  = sys.argv[2] if len(sys.argv) > 2 else "batch_results"
        card_type = sys.argv[3] if len(sys.argv) > 3 else "indian_passport"
        main(in_path, out_path, card_type)
