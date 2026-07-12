"""
download.py — Download candidate images collected by crawl.py.

Kept separate from crawling so you can re-download or re-filter without
re-crawling Pinterest. Dedupes by URL hash, skips anything already on disk,
rate-limited to be polite to Pinterest's CDN.

Usage:
    python src/download.py output/candidates.jsonl output/images
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

DELAY_SEC = 0.5
TIMEOUT_SEC = 15


def _filename_for(url: str) -> str:
    ext = Path(url.split("?")[0]).suffix or ".jpg"
    return hashlib.sha1(url.encode()).hexdigest()[:16] + ext


def main(candidates_file: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(candidates_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    manifest_path = out_path.parent / "download_manifest.jsonl"
    manifest = open(manifest_path, "a", encoding="utf-8")

    downloaded, skipped, failed = 0, 0, 0
    for rec in records:
        url = rec.get("image_url")
        if not url:
            continue
        fname = _filename_for(url)
        dst = out_path / fname
        if dst.exists():
            skipped += 1
            continue
        try:
            resp = requests.get(url, timeout=TIMEOUT_SEC, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            dst.write_bytes(resp.content)
            manifest.write(json.dumps({"file_name": fname, **rec}) + "\n")
            downloaded += 1
        except Exception as e:
            print(f"  failed {url}: {e}")
            failed += 1
        time.sleep(DELAY_SEC)

    manifest.close()
    print(f"Downloaded {downloaded}, skipped (already had) {skipped}, failed {failed}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
