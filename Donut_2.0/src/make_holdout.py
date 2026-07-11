"""
make_holdout.py — Carve a fixed held-out test set OUT of a verified metadata.jsonl,
before augmentation ever runs.

These images must never be trained on. Run this once per card type, right after
Label Studio verification and before augment.py.

Deterministic (seeded) selection — safe to re-run; if a holdout set already
exists for this card_type it will refuse to pull a second one.

Usage:
    python src/make_holdout.py data/real 10
    python src/make_holdout.py data/cod 10
"""
import json
import random
import shutil
import sys
from pathlib import Path

SEED = 42


def main(source_dir: str, n: int = 10):
    src = Path(source_dir)
    meta_path = src / "metadata.jsonl"
    with open(meta_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print(f"No records in {meta_path}.")
        sys.exit(1)
    if len(records) <= n:
        print(f"Only {len(records)} verified records in {meta_path} — need more than "
              f"{n} to hold out {n} and still have anything left to train on. Aborting.")
        sys.exit(1)

    card_type = records[0]["ground_truth"]["card_type"]
    out_dir = Path("data/holdout") / card_type
    existing_holdout = out_dir / "metadata.jsonl"
    if existing_holdout.exists():
        print(f"{existing_holdout} already exists — holdout set for '{card_type}' "
              f"was already carved out. Not pulling a second one.")
        sys.exit(0)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    shuffled = records[:]
    random.shuffle(shuffled)
    holdout = shuffled[:n]
    holdout_files = {r["file_name"] for r in holdout}
    remaining = [r for r in records if r["file_name"] not in holdout_files]

    moved, missing = 0, []
    for rec in holdout:
        src_img = src / "images" / rec["file_name"]
        dst_img = out_dir / "images" / rec["file_name"]
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
            moved += 1
        else:
            missing.append(rec["file_name"])

    with open(out_dir / "metadata.jsonl", "w") as f:
        for rec in holdout:
            f.write(json.dumps(rec) + "\n")

    with open(meta_path, "w") as f:
        for rec in remaining:
            f.write(json.dumps(rec) + "\n")

    print(f"Held out {len(holdout)} records for '{card_type}' -> {out_dir}")
    print(f"  images moved: {moved}")
    if missing:
        print(f"  WARNING: {len(missing)} image(s) not found, metadata moved anyway: {missing}")
    print(f"Remaining in {src}: {len(remaining)} (safe to run augment.py on)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
