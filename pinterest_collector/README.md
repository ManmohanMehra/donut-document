# Pinterest Card Data Collector

Feeds raw candidate images into `Donut_2.0/data/<card_type>/images/` (Phase 1
/ Phase 5 of that project's plan). This does **not** replace Label Studio
verification — everything collected here is unverified and goes through the
normal Gemini pre-label → manual review pipeline before it's training data.

**Why this exists instead of keyword search**: keyword search on Pinterest
turned out to be low-precision for this — opening a known-good ID card pin
and following "related pins" surfaced far more relevant results. So instead
of searching, this crawls outward from a small set of known-good seed pins
through Pinterest's own related-pins graph, then uses visual similarity
(DINOv2 embeddings) against your existing ~500 card samples to keep only
what actually looks like a card, and Gemini as a second opinion on the
borderline cases.

**Scraping backend**: [Crawlee](https://crawlee.dev/python/) (`crawlee`
PyPI package) — Apify's open-source, self-hosted Python crawling framework,
using its `PlaywrightCrawler` for the JS-heavy Pinterest frontend. Free, no
Apify account or paid actor needed.

## Pipeline

```
seeds/seed_urls.txt          (you fill this in — known-good pin URLs)
        |
        v
   src/crawl.py        --> output/candidates.jsonl   (pin URLs + image URLs, BFS via related pins)
        |
        v
  src/download.py       --> output/images/ + output/download_manifest.jsonl
        |
        v
src/filter_similarity.py --> output/filtered/{keep,borderline}/  (vs seed_index.npz)
        |
        v
classify_gemini.py (on borderline only) --> output/classified.jsonl
        |
        v
  you manually promote "keep" + confidently-classified "borderline" images
  into Donut_2.0/data/<card_type>/images/, then run the normal Phase 1
  Gemini pre-label + Label Studio verification from there.
```

## Setup

```bash
conda activate donut2       # or whatever env has Donut_2.0/requirements.txt installed
pip install -r requirements.txt
playwright install chromium
```

## One-time: build the seed similarity index

Point this at wherever your ~500 known-good card samples live:

```bash
python src/build_seed_index.py ../Donut_2.0/data/passport_data seed_index.npz
```

## One-time (optional but recommended): logged-in session

Related-pins tends to be richer when logged in, and un-authenticated
scraping is rate-limited harder. Log in once by hand and save the session —
**`storage_state.json` contains your login cookies, it is gitignored, never
commit it**:

```bash
playwright codegen --save-storage=storage_state.json https://www.pinterest.com
# log in in the window that opens, then close it
```

## Run

```bash
# add known-good pin URLs to seeds/seed_urls.txt first
python src/pipeline.py seeds/seed_urls.txt seed_index.npz
```

Then, for the borderline bucket only:

```bash
export GOOGLE_API_KEY=AIza...   # never paste it into a file
python src/classify_gemini.py output/filtered/borderline output/filtered/borderline.jsonl output/classified.jsonl
```

## Known-fragile part

`src/crawl.py`'s page parser (`_extract_pins_from_page`) depends on
Pinterest embedding pin/image data as JSON in the page — this isn't a
documented public API and the exact structure has shifted before. If a
crawl run returns 0 candidates, open a pin in a real browser, devtools >
Elements, search for `"orig"`, and adjust the parser to match what you
actually see. There's a regex fallback for raw `i.pinimg.com/originals/`
URLs if the structured parse comes up empty, but it won't get you
related-pin traversal, just whatever's already on the page.

## Ground rules

- **Pinterest's ToS restricts automated scraping.** This is scoped for
  internal CV R&D data collection at modest volume (`MAX_PINS = 500` per run
  in `crawl.py`), not bulk harvesting or redistribution. Don't raise that
  cap without thinking about why.
- Everything downloaded here is **unverified** — it goes through Phase 1's
  normal Gemini pre-label + manual Label Studio review before it's training
  data, same as any other source.
- `output/`, `seed_index.npz`, and `storage_state.json` are gitignored —
  none of it gets pushed to GitHub.
