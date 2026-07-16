"""
build_multi_type_dataset.py — Merge every verified per-type dataset into
data/multi_type/. Auto-discovers sources instead of a hardcoded list, since
schemas.py now supports 100+ card types and hand-maintaining a source dict
per type doesn't scale.

Run this AFTER a type is labeled and converted to metadata.jsonl — it just
picks up whatever's there. Copies images and builds a single interleaved
metadata.jsonl for training.

Usage:
    python src/build_multi_type_dataset.py

Discovery rule: any data/<name>/metadata.jsonl found, where images referenced
by file_name live in data/<name>/images/. card_type comes from each record's
own ground_truth.card_type (not the folder name), so folder naming doesn't
need to match schemas.py exactly. Unknown card_types (typos, or a type
retired from schemas.py) are flagged, not silently dropped.

Output:
    data/multi_type/
        metadata.jsonl   (all types, shuffled)
        images/          (copied from per-type folders)
"""

import json
import random
import shutil
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from schemas import SUPPORTED_CARD_TYPES

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "multi_type"
SEED = 42


def main():
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    all_records = []
    unknown_types = Counter()

    for meta_path in sorted(DATA_DIR.glob("*/metadata.jsonl")):
        src_dir = meta_path.parent
        if src_dir == OUT_DIR:
            continue

        with open(meta_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        img_src_dir = src_dir / "images"
        copied = 0
        skipped = 0

        for rec in records:
            rec.pop("_review_flag", None)  # strip any pending-review marker before training

            card_type = rec.get("ground_truth", {}).get("card_type")
            if card_type not in SUPPORTED_CARD_TYPES:
                unknown_types[f"{card_type!r} (in {meta_path})"] += 1
                continue

            src_img = img_src_dir / rec["file_name"]
            dst_img = OUT_DIR / "images" / rec["file_name"]
            if not src_img.exists():
                skipped += 1
                continue

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            all_records.append(rec)
            copied += 1

        print(f"  {src_dir.name}: {copied} records added ({skipped} skipped — image not found)")

    if unknown_types:
        print("\nSkipped records with a card_type not in schemas.py:")
        for label, n in unknown_types.items():
            print(f"  {label}: {n}")

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
