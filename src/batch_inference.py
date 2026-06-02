import os
import json
import csv
import time
from pathlib import Path
from tqdm import tqdm
from inference import load_model, run_inference

def find_images(directory):
    """Find all images in nested folders."""
    list_of_images = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                list_of_images.append(os.path.join(root, file))
    return list_of_images

def main(input_path: str, output_dir: str = "batch_results"):
    # 1. Setup
    os.makedirs(output_dir, exist_ok=True)
    images = find_images(input_path)
    if not images:
        print(f"No images found in {input_path}")
        return

    print(f"🚀 Found {len(images)} images. Loading model...")
    model, processor, device = load_model()
    
    results = []
    
    # 2. Process
    print(f"📸 Starting batch processing...")
    for img_path in tqdm(images):
        try:
            # We pass the pre-loaded model/processor here for speed
            result = run_inference(img_path, model=model, processor=processor, device=device)
            
            # Add metadata about the file
            entry = {
                "filename": os.path.basename(img_path),
                "relative_path": os.path.relpath(img_path, input_path),
                "prediction": result.get("indian_passport", {}),
                "execution_time_sec": result.get("execution_time_sec")
            }
            results.append(entry)
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")

    # 3. Save JSON (The complete data)
    json_output = os.path.join(output_dir, "results.json")
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 4. Save CSV (The audit-friendly data)
    csv_output = os.path.join(output_dir, "results.csv")
    if results:
        # Get all unique field names from the passport results
        fieldnames = ["filename", "relative_path", "execution_time_sec"]
        all_keys = set()
        for r in results:
            all_keys.update(r["prediction"].keys())
        fieldnames.extend(sorted(list(all_keys)))

        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "filename": r["filename"],
                    "relative_path": r["relative_path"],
                    "execution_time_sec": r["execution_time_sec"]
                }
                # Flatten the prediction dict into the row
                for k, v in r["prediction"].items():
                    # Convert dicts like mrz_validation to string for CSV
                    row[k] = str(v) if isinstance(v, (dict, list)) else v
                writer.writerow(row)

    print(f"\n✅ Done! Processed {len(images)} images.")
    print(f"📁 JSON: {json_output}")
    print(f"📁 CSV:  {csv_output}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/batch_inference.py <input_folder_path> [output_folder_name]")
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else "batch_results"
        main(in_path, out_path)
