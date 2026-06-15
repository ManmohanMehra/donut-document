import os
import json
from pathlib import Path
from tqdm import tqdm
from inference import load_model, run_inference

def main(input_dir: str, output_file: str):
    # 1. Setup
    input_path = Path(input_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in os.listdir(input_path) if Path(f).suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"🤖 Loading current best model to auto-label {len(images)} images...")
    model, processor, device = load_model()
    
    # 2. Process
    with open(output_file, "w", encoding="utf-8") as f:
        for img_name in tqdm(images):
            img_path = str(input_path / img_name)
            try:
                # Use current model to suggest labels
                result = run_inference(img_path, model=model, processor=processor, device=device)
                
                # Format for Donut metadata.jsonl
                # The prompt is important for training
                gt_json = result.get("indian_passport", {})
                
                record = {
                    "file_name": img_name,
                    "ground_truth": json.dumps({"gt_parse": gt_json}, ensure_ascii=False)
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"\nError labeling {img_name}: {e}")

    print(f"\n✅ Done! Suggested labels saved to: {output_file}")
    print("👉 Next step: Open this file, check the data, and copy it to your main metadata.jsonl")

if __name__ == "__main__":
    main("data/IND", "data/IND/metadata_suggested.jsonl")
