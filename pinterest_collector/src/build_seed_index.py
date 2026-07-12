"""
build_seed_index.py — Embed your known-good card samples once, save the index.

The crawl/filter steps reuse this index instead of re-embedding the seed set
every run.

Usage:
    python src/build_seed_index.py /path/to/500_card_samples seed_index.npz
    python src/build_seed_index.py ../Donut_2.0/data/real/images seed_index.npz

seed_dir is searched recursively, so you can point it at a folder of
per-card-type subfolders (e.g. Donut_2.0/data/passport_data/) and it will
pick up every image underneath.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from embed import Embedder

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main(seed_dir: str, out_path: str):
    seed_path = Path(seed_dir)
    images = sorted(p for p in seed_path.rglob("*") if p.suffix.lower() in EXTS)
    if not images:
        print(f"No images found under {seed_dir}")
        sys.exit(1)

    print(f"Embedding {len(images)} seed images with DINOv2...")
    embedder = Embedder()
    vectors, names = [], []
    for i, img_path in enumerate(images):
        try:
            vectors.append(embedder.embed_image(str(img_path)))
            names.append(str(img_path.relative_to(seed_path)))
        except Exception as e:
            print(f"  skipping {img_path.name}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)}")

    np.savez(out_path, vectors=np.stack(vectors), names=np.array(names))
    print(f"Saved {len(names)} seed embeddings -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
