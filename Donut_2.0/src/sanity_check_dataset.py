"""
sanity_check_dataset.py — Phase 2 step 5: verify a metadata.jsonl is actually
ready to train on before spending GPU time on it.

Checks:
  - every image referenced in metadata.jsonl exists on disk
  - every token used by the ground_truth (<s_card_type>, <s_field>, ...) is
    present in the processor's saved vocabulary — this is exactly the kind of
    mismatch that happens when a card type is added to schemas.py but
    add_tokens.py hasn't been rerun, and it fails loudly during training
    rather than silently at inference time
  - prints a few real token sequences so you can eyeball the format

Usage:
    python src/sanity_check_dataset.py data/multi_type
    python src/sanity_check_dataset.py data/augmented
"""
import json
import random
import sys
from pathlib import Path

from transformers import DonutProcessor


def gt_to_token_sequence(gt: dict) -> str:
    gt = gt.copy()
    card_type = gt.pop("card_type")
    seq = f"<s_{card_type}>"
    for key, value in gt.items():
        seq += f"<s_{key}>{value or ''}</s_{key}>"
    seq += f"</s_{card_type}>"
    return seq


def main(data_dir: str, num_samples: int = 3, processor_path: str = "checkpoints/donut-passport-processor"):
    data_path = Path(data_dir)
    meta_path = data_path / "metadata.jsonl"
    with open(meta_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"{meta_path}: {len(records)} records")

    missing_images = [r["file_name"] for r in records if not (data_path / "images" / r["file_name"]).exists()]
    if missing_images:
        print(f"  {len(missing_images)} record(s) reference missing images, e.g. {missing_images[:5]}")
    else:
        print("  all referenced images exist")

    vocab = None
    try:
        processor = DonutProcessor.from_pretrained(processor_path, local_files_only=True)
        vocab = set(processor.tokenizer.additional_special_tokens)
    except Exception as e:
        print(f"  Could not load processor at {processor_path} ({e}) — skipping vocab check")

    print(f"\nSample token sequences ({min(num_samples, len(records))} of {len(records)}):")
    random.seed(0)
    for rec in random.sample(records, min(num_samples, len(records))):
        print(f"\n--- {rec['file_name']} ---")
        print(gt_to_token_sequence(rec["ground_truth"]))

    if vocab is not None:
        unknown = set()
        for rec in records:
            gt = rec["ground_truth"]
            card_type = gt["card_type"]
            unknown |= {f"<s_{card_type}>", f"</s_{card_type}>"} - vocab
            for key in gt:
                if key == "card_type":
                    continue
                unknown |= {f"<s_{key}>", f"</s_{key}>"} - vocab
        if unknown:
            print(f"\n{len(unknown)} token(s) used in this data are NOT in the processor vocab:")
            print(f"  {sorted(unknown)}")
            print("  -> rerun add_tokens.py (schemas.py has fields this processor was never taught).")
        else:
            print("\nAll tokens used in this data are present in the processor vocab. Good to train.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
