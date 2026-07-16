"""
pre_annotate_all.py — Run pre-annotation across every card-type folder in
data/passport_data/ in one go, instead of typing one command per folder for
100+ types. Supports three interchangeable backends producing identical
output format:
  - gemini   (vision_label.py)          — Gemini API, needs GOOGLE_API_KEY, rate-limited
  - ollama   (pre_annotate_ollama.py)   — local MiniCPM-V via Ollama, free, no rate limit,
                                           confirmed to run on Apple Silicon (recommended for your Mac)
  - minicpm  (pre_annotate_minicpm.py)  — local MiniCPM-V via raw transformers, needs a real GPU
                                           (A40 server) — Apple Silicon support unconfirmed for this path

Folder → card_type mapping (same convention as scripts/rename.py and
schemas.py): bare 3-letter alpha code = "{code}_passport", anything else =
lowercased with hyphens → underscores. Special case: IND → indian_passport
(NOT ind_passport), so new IND labels land in the same card_type as the 19
already-verified records in data/real — otherwise the model would learn two
competing tokens for the same document.

Deliberately skipped:
  - Junk folders: ANIMAL, ANIME, CELEB, BLANK, MIX (not ID documents)
  - Nested subfolders (IND/FULL, IND/back, */RAW, */TAMPERED, OMN-DL/OMN):
    these carry meaning — TAMPERED especially must never be labeled as a
    genuine document — so only each folder's top-level images are processed.
    Nested folders are listed at the end for you to decide about.
  - Folders whose card_type isn't in schemas.py (fails loudly, listed at end)

Output: data/passport_data/<FOLDER>/metadata_suggested.jsonl per folder.
Already-done folders (existing non-empty metadata_suggested.jsonl) are
skipped, so the run is resumable — rate limits, a dropped connection, or a
killed GPU job just mean running it again.

Usage:
    export GOOGLE_API_KEY=AIza...   # terminal only, never in a file — gemini backend only
    python src/pre_annotate_all.py                             # gemini, all folders
    python src/pre_annotate_all.py --backend ollama             # local via Ollama, all folders
    python src/pre_annotate_all.py --backend ollama IND ZWE PAK  # local, only these folders
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schemas import SUPPORTED_CARD_TYPES

DATA_DIR = Path("data/passport_data")
JUNK = {"ANIMAL", "ANIME", "CELEB", "BLANK", "MIX"}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BACKENDS = {"gemini", "ollama", "minicpm"}
_BACKEND_MODULES = {
    "gemini": "vision_label",
    "ollama": "pre_annotate_ollama",
    "minicpm": "pre_annotate_minicpm",
}


def folder_to_card_type(name: str) -> str:
    if name == "IND":
        return "indian_passport"
    if len(name) == 3 and name.isalpha():
        return f"{name.lower()}_passport"
    return name.replace("-", "_").lower()


def _parse_args(argv: list[str]) -> tuple[str, list[str]]:
    backend = "gemini"
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--backend":
            if i + 1 >= len(argv) or argv[i + 1] not in BACKENDS:
                print(f"--backend needs one of {sorted(BACKENDS)}")
                sys.exit(1)
            backend = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return backend, rest


def main(argv: list[str]):
    backend, only = _parse_args(argv)
    labeler = __import__(_BACKEND_MODULES[backend])

    if not DATA_DIR.exists():
        print(f"{DATA_DIR} not found — run from the Donut_2.0 root.")
        sys.exit(1)

    todo, skipped_done, unknown, nested = [], [], [], []

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or d.name in JUNK:
            continue
        if only and d.name not in only:
            continue

        for sub in sorted(p.name for p in d.iterdir() if p.is_dir()):
            nested.append(f"{d.name}/{sub}")

        n_images = sum(1 for f in d.iterdir() if f.suffix.lower() in EXTS)
        if n_images == 0:
            continue

        card_type = folder_to_card_type(d.name)
        if card_type not in SUPPORTED_CARD_TYPES:
            unknown.append((d.name, card_type))
            continue

        out_file = d / "metadata_suggested.jsonl"
        # A zero-byte file means a previous run errored on every image
        # (e.g. bad API key) — treat that as not done, not as complete.
        if out_file.exists() and out_file.stat().st_size > 0:
            skipped_done.append(d.name)
            continue

        todo.append((d, out_file, card_type, n_images))

    total_images = sum(n for *_, n in todo)
    print(f"backend: {backend}")
    print(f"{len(todo)} folder(s) to label, ~{total_images} images")
    if skipped_done:
        print(f"{len(skipped_done)} folder(s) already have metadata_suggested.jsonl — skipped (delete the file to redo)")

    for i, (d, out_file, card_type, n_images) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {d.name} -> {card_type} ({n_images} images)")
        labeler.main(str(d), str(out_file), card_type)

    if unknown:
        print("\nFolders whose card_type is NOT in schemas.py (add it there, then rerun):")
        for name, ct in unknown:
            print(f"  {name} -> {ct}")
    if nested:
        print("\nNested subfolders NOT processed (decide what these are before labeling):")
        for n in nested:
            print(f"  {n}")


if __name__ == "__main__":
    main(sys.argv[1:])
