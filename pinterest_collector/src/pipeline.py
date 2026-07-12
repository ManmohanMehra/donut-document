"""
pipeline.py — Run the full collection pipeline end to end with default paths.

Equivalent to running crawl.py -> download.py -> filter_similarity.py in
sequence. classify_gemini.py is intentionally NOT included here — it needs
your GOOGLE_API_KEY and costs API calls, so run it as a separate, deliberate
step on output/filtered/borderline.jsonl once you've eyeballed a sample of
the "keep" bucket and trust the similarity threshold.

Usage:
    python src/pipeline.py seeds/seed_urls.txt seed_index.npz
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import crawl
import download
import filter_similarity


def main(seed_file: str, seed_index_file: str, out_dir: str = "output"):
    out_path = Path(out_dir)
    candidates_file = out_path / "candidates.jsonl"
    images_dir = out_path / "images"
    manifest_file = out_path / "download_manifest.jsonl"
    filtered_dir = out_path / "filtered"

    print("=== 1/3 Crawling related-pins graph ===")
    asyncio.run(crawl.run(seed_file, str(candidates_file)))

    print("\n=== 2/3 Downloading candidates ===")
    download.main(str(candidates_file), str(images_dir))

    print("\n=== 3/3 Filtering by visual similarity to seed set ===")
    filter_similarity.main(str(manifest_file), str(images_dir), seed_index_file, str(filtered_dir))

    print(f"\nDone. Review {filtered_dir}/keep before promoting into Donut_2.0/data/<card_type>/images/.")
    print(f"Borderline cases are in {filtered_dir}/borderline — run classify_gemini.py on those next.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:4])
