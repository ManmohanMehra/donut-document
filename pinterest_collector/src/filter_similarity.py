"""
filter_similarity.py — Score downloaded candidates against the seed index and
bucket them into keep / borderline / discard.

This replaces manual eyeballing of every crawled image with a tunable
precision/recall knob. Borderline cases go to classify_gemini.py for a
second opinion instead of being auto-discarded.

Thresholds are starting points, not calibrated values — after a first run,
look at a sample of "keep" and "discard" images and adjust KEEP_THRESHOLD /
BORDERLINE_THRESHOLD in a config, or pass them as CLI args.

Usage:
    python src/filter_similarity.py output/download_manifest.jsonl output/images seed_index.npz output/filtered
    python src/filter_similarity.py output/download_manifest.jsonl output/images seed_index.npz output/filtered 0.8 0.6
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embed import Embedder, best_match

KEEP_THRESHOLD = 0.75
BORDERLINE_THRESHOLD = 0.55


def main(
    manifest_file: str,
    images_dir: str,
    seed_index_file: str,
    out_dir: str,
    keep_threshold: float = KEEP_THRESHOLD,
    borderline_threshold: float = BORDERLINE_THRESHOLD,
):
    images_path = Path(images_dir)
    out_path = Path(out_dir)
    (out_path / "keep").mkdir(parents=True, exist_ok=True)
    (out_path / "borderline").mkdir(parents=True, exist_ok=True)

    seed = np.load(seed_index_file)
    seed_vectors = seed["vectors"]

    with open(manifest_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    embedder = Embedder()
    results = {"keep": [], "borderline": [], "discard": []}

    for i, rec in enumerate(records):
        img_path = images_path / rec["file_name"]
        if not img_path.exists():
            continue
        try:
            vec = embedder.embed_image(str(img_path))
        except Exception as e:
            print(f"  skipping {rec['file_name']}: {e}")
            continue
        _, sim = best_match(vec, seed_vectors)
        rec["similarity"] = sim

        if sim >= keep_threshold:
            bucket = "keep"
        elif sim >= borderline_threshold:
            bucket = "borderline"
        else:
            bucket = "discard"
        results[bucket].append(rec)

        if bucket in ("keep", "borderline"):
            shutil.copy2(img_path, out_path / bucket / rec["file_name"])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)}")

    for bucket, recs in results.items():
        if bucket == "discard":
            continue
        with open(out_path / f"{bucket}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")

    print(f"\nkeep: {len(results['keep'])}  borderline: {len(results['borderline'])}  discard: {len(results['discard'])}")
    print(f"Next: run classify_gemini.py on {out_path / 'borderline.jsonl'} to sort the borderline ones by card type.")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    args = sys.argv[1:5]
    thresholds = [float(x) for x in sys.argv[5:7]] if len(sys.argv) > 5 else []
    main(*args, *thresholds)
