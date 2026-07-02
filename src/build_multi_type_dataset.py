"""
build_multi_type_dataset.py — Merge per-type labeled data into data/multi_type/

Run this AFTER all 4 types are labeled and converted to metadata.jsonl.
Copies images and builds a single interleaved metadata.jsonl for training.

Usage:
    python src/build_multi_type_dataset.py

Expected input layout:
    data/real/           metadata.jsonl + images/   ← IND (already labeled)
    data/are_fed_card/   metadata.jsonl + images/
    data/cod/            metadata.jsonl + images/
    data/zwe/            metadata.jsonl + images/

Output:
    data/multi_type/
        metadata.jsonl   (all types, shuffled)
        images/          (symlinked or copied from per-type folders)
"""

import json
import random
import shutil
from pathlib import Path
from collections import Counter

# Map card_type → source data directory
SOURCES = {
    "indian_passport": Path("data/real"),
    "are_fed_card":    Path("data/are_fed_card"),
    "cod_passport":    Path("data/cod"),
    "zwe_passport":    Path("data/zwe"),
}

OUT_DIR = Path("data/multi_type")
SEED = 42


def main():
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    all_records = []
    missing = []

    for card_type, src_dir in SOURCES.items():
        meta_path = src_dir / "metadata.jsonl"
        if not meta_path.exists():
            missing.append(str(meta_path))
            print(f"  MISSING: {meta_path}")
            continue

        with open(meta_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        img_src_dir = src_dir / "images"
        copied = 0
        skipped = 0

        for rec in records:
            # Strip any _review_flag added by vision_label.py before including in training
            rec.pop("_review_flag", None)

            src_img = img_src_dir / rec["file_name"]
            dst_img = OUT_DIR / "images" / rec["file_name"]

            if not src_img.exists():
                skipped += 1
                continue

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            all_records.append(rec)
            copied += 1

        print(f"  {card_type}: {copied} records added ({skipped} skipped — image not found)")

    if missing:
        print(f"\nWarning: {len(missing)} metadata file(s) missing — run labeling for those types first.")

    # Interleave: shuffle with fixed seed for reproducibility
    random.seed(SEED)
    random.shuffle(all_records)

    out_meta = OUT_DIR / "metadata.jsonl"
    with open(out_meta, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = Counter(r["ground_truth"]["card_type"] for r in all_records)
    print(f"\nBuilt {out_meta}")
    print(f"Total: {len(all_records)} records")
    for ct, n in sorted(counts.items()):
        print(f"  {ct}: {n}")


if __name__ == "__main__":
    main()
