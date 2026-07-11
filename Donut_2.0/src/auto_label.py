import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from inference import load_model, run_inference
from schemas import SUPPORTED_CARD_TYPES


def main(input_dir: str, output_file: str, card_type: str):
    """
    Use the current best checkpoint to generate suggested labels for a folder of images.
    Output is a metadata_suggested.jsonl — review and correct before adding to training data.

    Args:
        input_dir:   folder of raw images for one card type
        output_file: path to write suggested labels (metadata_suggested.jsonl)
        card_type:   one of SUPPORTED_CARD_TYPES
    """
    if card_type not in SUPPORTED_CARD_TYPES:
        print(f"Unknown card_type '{card_type}'. Supported: {SUPPORTED_CARD_TYPES}")
        sys.exit(1)

    input_path = Path(input_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in os.listdir(input_path) if Path(f).suffix.lower() in image_extensions]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Loading model to auto-label {len(images)} images as '{card_type}'...")
    model, processor, device = load_model()

    with open(output_file, "w", encoding="utf-8") as f:
        for img_name in tqdm(images):
            img_path = str(input_path / img_name)
            try:
                result = run_inference(img_path, card_type=card_type, model=model, processor=processor, device=device)
                gt_json = result.get(card_type, {})
                gt_json.pop("mrz_validation", None)  # validation result is not a label field
                record = {
                    "file_name": img_name,
                    "ground_truth": {"card_type": card_type, **gt_json},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\nError labeling {img_name}: {e}")

    print(f"\nDone! Suggested labels saved to: {output_file}")
    print("Next step: review and correct in Label Studio, then convert with convert_labelstudio.py")


if __name__ == "__main__":
    # Usage: python src/auto_label.py <image_dir> <output.jsonl> <card_type>
    # Examples:
    #   python src/auto_label.py data/cod/images  data/cod/metadata_suggested.jsonl  cod_passport
    #   python src/auto_label.py data/zwe/images  data/zwe/metadata_suggested.jsonl  zwe_passport
    if len(sys.argv) != 4:
        print("Usage: python src/auto_label.py <image_dir> <output.jsonl> <card_type>")
        print(f"Supported card types: {SUPPORTED_CARD_TYPES}")
        sys.exit(1)

    main(
        input_dir=sys.argv[1],
        output_file=sys.argv[2],
        card_type=sys.argv[3],
    )
