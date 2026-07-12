"""
crawl.py — BFS Pinterest's "related pins" graph starting from known-good seed
pins, instead of using Pinterest's keyword search box (which turned out to
surface far less relevant results for this than opening a good pin and
following "More like this").

Flow:
    seed pin URLs (seeds/seed_urls.txt)
      -> open each pin page in a headless browser
      -> parse the page for the pin's own full-res image + related pins
      -> enqueue newly-seen pin URLs, repeat up to MAX_DEPTH / MAX_PINS
      -> write every seen pin's image URL + pin URL to output/candidates.jsonl
         (no images downloaded yet — that's download.py, kept separate so a
         crawl can be re-filtered without re-crawling)

FRAGILE PART: `_extract_pins_from_page`. Pinterest embeds full pin data
(including the original-resolution image URL) as JSON inside a <script
type="application/json"> tag on each pin page. The exact structure has
changed before and isn't a documented public API — this looks for any such
script tag whose JSON contains an "images"/"orig" pair rather than
hardcoding a specific script id, to survive minor markup changes, and falls
back to a raw regex scan for i.pinimg.com/originals/ URLs if that fails. If
you're getting zero results, open a pin in a real browser, devtools >
Elements, search for "orig", and adjust the parser here to match what you
see.

Pinterest ToS restricts automated scraping — this defaults to a modest pace
and a hard cap on pins per run. Treat this as internal CV R&D data
collection, not something to run unattended at scale or redistribute.

Setup:
    pip install crawlee[playwright]
    playwright install chromium

Optional logged-in session (related pins are often richer when logged in):
    Run once interactively to save a session, e.g. via
    `playwright codegen --save-storage=storage_state.json https://www.pinterest.com`,
    log in by hand in the opened browser, then close it. crawl.py will pick
    up storage_state.json automatically if it exists in this directory.

Usage:
    python src/crawl.py seeds/seed_urls.txt output/candidates.jsonl
"""
import asyncio
import json
import re
import sys
from pathlib import Path

from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee import Request

MAX_PINS = 500                     # hard cap per run — keep volume modest
MAX_DEPTH = 3                      # BFS hops from a seed pin
REQUEST_DELAY_SEC = 3.0            # politeness floor between pin loads
STORAGE_STATE_PATH = Path(__file__).parent.parent / "storage_state.json"


def _walk_json_for_pins(node, found):
    """Recursively find dict nodes that look like a Pinterest 'pin' object
    (has an "images" dict with an "orig" variant)."""
    if isinstance(node, dict):
        images = node.get("images")
        if isinstance(images, dict) and isinstance(images.get("orig"), dict):
            pin_id = node.get("id")
            found.append({
                "image_url": images["orig"].get("url"),
                "pin_url": f"https://www.pinterest.com/pin/{pin_id}/" if pin_id else None,
                "pin_id": pin_id,
            })
        for v in node.values():
            _walk_json_for_pins(v, found)
    elif isinstance(node, list):
        for v in node:
            _walk_json_for_pins(v, found)


def _extract_pins_from_page(html: str) -> list[dict]:
    found = []
    for raw in re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.DOTALL):
        if '"orig"' not in raw or '"images"' not in raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        _walk_json_for_pins(data, found)

    if not found:
        # Fallback: raw scan for full-resolution image URLs on the page.
        urls = set(re.findall(
            r'https://i\.pinimg\.com/originals/[^\s"\'\\]+\.(?:jpg|jpeg|png|webp)', html
        ))
        found = [{"image_url": u, "pin_url": None, "pin_id": None} for u in urls]

    # de-dupe by image_url within this single page's extraction
    seen, unique = set(), []
    for pin in found:
        if pin["image_url"] and pin["image_url"] not in seen:
            seen.add(pin["image_url"])
            unique.append(pin)
    return unique


async def run(seed_file: str, out_file: str):
    seed_urls = [
        line.strip() for line in Path(seed_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not seed_urls:
        print(f"No seed URLs in {seed_file} — add known-good pin URLs first (one per line).")
        sys.exit(1)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w", encoding="utf-8")

    seen_pin_ids: set[str] = set()
    seen_image_urls: set[str] = set()
    state = {"count": 0}

    launch_options = {"storage_state": str(STORAGE_STATE_PATH)} if STORAGE_STATE_PATH.exists() else {}
    crawler = PlaywrightCrawler(
        max_requests_per_crawl=MAX_PINS,
        headless=True,
        browser_launch_options=launch_options,
    )

    @crawler.router.default_handler
    async def handler(context: PlaywrightCrawlingContext) -> None:
        if state["count"] >= MAX_PINS:
            return
        depth = context.request.user_data.get("depth", 0)
        context.log.info(f"[depth {depth}] {context.request.url}")

        await asyncio.sleep(REQUEST_DELAY_SEC)
        html = await context.page.content()
        pins = _extract_pins_from_page(html)

        next_requests = []
        for pin in pins:
            if state["count"] >= MAX_PINS:
                break
            key = pin.get("pin_id") or pin["image_url"]
            if key in seen_pin_ids or pin["image_url"] in seen_image_urls:
                continue
            seen_pin_ids.add(key)
            seen_image_urls.add(pin["image_url"])

            out_f.write(json.dumps({**pin, "found_via": context.request.url, "depth": depth}) + "\n")
            out_f.flush()
            state["count"] += 1

            if depth < MAX_DEPTH and pin.get("pin_url"):
                next_requests.append(Request.from_url(pin["pin_url"], user_data={"depth": depth + 1}))

        if next_requests:
            await context.add_requests(next_requests)

    await crawler.run([Request.from_url(u, user_data={"depth": 0}) for u in seed_urls])
    out_f.close()
    print(f"Collected {state['count']} candidate pins -> {out_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    asyncio.run(run(sys.argv[1], sys.argv[2]))
