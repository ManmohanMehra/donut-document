# Donut 2.0 — Status Tracker

Honest numbers as of 2026-07-13.

## Major change: unified schema (Path B)

Per the Gemini/Claude R&D chat you shared, this project moved from
"passport parser + a few more passport types" to a generalized multi-type
document parser — 104 real document types surfaced from
`~/Downloads/passport_data` (passports, national IDs, driving licences,
residence permits, civil IDs across ~60 countries), not the 3-4 types
`next_steps_action_plan.md` and the old `LABELING_GUIDE.md` were written
against.

`schemas.py` was rewritten around **one shared field set** (`document_number`,
`surname`, `given_names`, `dob`, `sex`, `nationality`, `date_of_issue`,
`date_of_expiry`, `place_of_birth`, `mrz_line1`, `mrz_line2`) used by all 106
card types, instead of a hand-written field list per type. This was a
deliberate tradeoff you chose over "unified core + per-type extras" —
type-specific fields from the old per-type schemas (`place_of_issue`, `type`
P/S/D/O, `country_code`, `emirates_id`/`issuing_authority`) were dropped, not
kept as optional extras. See `schemas.py`'s docstring for the full reasoning
and the folder-name → card_type naming convention.

**Consequence for existing verified data**: the 19 verified `indian_passport`
records were migrated — `passport_number` → `document_number`, and `type`
/ `country_code` / `place_of_issue` were dropped (not silently invented
elsewhere). `inference.py`'s MRZ backfill was updated to match
(`document_number` instead of `passport_number`). `build_multi_type_dataset.py`
was rewritten to auto-discover `data/*/metadata.jsonl` instead of a
hardcoded 4-entry source list, since that doesn't scale to 106 types.

## Bug I found and fixed this session

The very first `Donut_2.0/data/real/` copy silently failed (a transient tool
error) and was never retried — `STATUS.md` and this whole tracker were
describing a folder that didn't exist. Re-copied from the root `data/real`
(untouched, still 20 images / 19 records) and re-ran the schema migration
against the real files this time.

## Unresolved — needs your input, not touched

`data/passport_data/` — in **both** `Donut_2.0/data/` and the original root
`data/passport_data/` — now contains the full 110-folder structure mirroring
`~/Downloads/passport_data` (including a `FULL` subfolder inside `IND/` that
wasn't part of anything copied in this session). This wasn't done by me in
this conversation. Since `data/` is gitignored, I can't tell from git history
what put it there (manual copy, iCloud/Dropbox sync, another tool). I left it
alone rather than guess and possibly overwrite something you're mid-way
through organizing — flag when you know what should happen to it.

## Status table

| Task | Status | Notes |
|---|---|---|
| Indian passport samples verified (`data/real`) | 19 verified, migrated to unified schema, 1 orphan image | `pass_21.jpeg` has no metadata record — needs labeling or removal |
| Held-out test set (10 images, never trained on) | Not carved out yet | Run `make_holdout.py` before any augmentation |
| Raw multi-type collection | **734 images across 104 document types**, in `~/Downloads/passport_data` (outside the repo) | Named per `scripts/rename.py`'s convention; junk folders `ANIMAL`/`ANIME`/`CELEB`/`BLANK`/`MIX` and loose `MIX_PASSPORT_*.jpg` excluded from the 104 |
| Data collection style guidance | Documented here, not yet in a README | Natural variation preferred over tight crops: card in hand/on table, slight angle, some background all fine; avoid extreme crops into text fields, backgrounds so large the card is <40% of frame, or pure black/white backgrounds. Augmentation pipeline already assumes this. |
| `schemas.py` | **Rewritten — unified schema, 106 card types, 234 tokens** | See above |
| Pre-annotation pipeline | Code ready, 3 backends via `pre_annotate_all.py --backend <name>` | `gemini` (`vision_label.py`) hit 401→429→503 in practice — free tier deprioritizes traffic under load and has hard daily caps, not viable at 800-20K scale without enabling billing (declined). `ollama` (`pre_annotate_ollama.py`) is the recommended path: local MiniCPM-V-4.6 via Ollama, confirmed Apple Silicon support, free, no rate limit — needs `brew install ollama && ollama serve && ollama pull minicpm-v4.6` first. `minicpm` (`pre_annotate_minicpm.py`, raw transformers) is the fallback for the A40 server if Ollama proves too slow at scale. |
| Label Studio verification | Not started for any of the 104 new types | IND (19) was verified manually in iteration 1, under the old schema — now migrated |
| Augmentation pipeline | Code ready (`augment.py`) | Last real run (iteration 1) produced 598 images from the pre-holdout 19; needs rerunning after holdout split |
| Token registration | **Stale — must rerun** | `checkpoints/donut-passport-processor` still has the old 19 iteration-1 tokens. Needs all 234 current tokens. Run `add_tokens.py`. |
| Dataset sanity check | Script ready (`sanity_check_dataset.py`) | Run after `add_tokens.py` + `build_multi_type_dataset.py`, before every training run |
| `build_multi_type_dataset.py` | **Rewritten — auto-discovers `data/*/metadata.jsonl`** | No longer a hardcoded 4-type list; will pick up any type as soon as it has verified data |
| First Donut training run (A40) | Prior run exists (`checkpoints/donut-passport-final`) — predates unified schema entirely | Retrain needed once any real multi-type data + refreshed tokens are ready |
| POC inference + validation | Code ready (`inference.py`, `batch_inference.py`, `mrz_validator.py`) | Blocked on a checkpoint trained with the current tokenizer |

## Known bad rows in `data/real/metadata.jsonl` (carried over as-is, not edited)

Per the plan's own rule ("MRZ lines — always verify manually, never trust
blindly"), these were not hand-fixed since the source images weren't
re-checked. Field names below are post-migration:

- `pass_2.png` — `date_of_expiry: "23705/2031"` (malformed, extra digit)
- `pass_12.jpg` — `document_number: "Ԣ5101066"` (corrupted non-ASCII char, should likely be `T`); `mrz_line1` has a stray `C` before `HARWANT`; `mrz_line2` ends in `O` where a digit is expected
- `pass_10.jpg` (S1157960 record, `pass_22.jpg`) — `mrz_line2` ends in lowercase `o` where a digit is expected
- `pass_21.jpeg` — image present, no metadata record at all

## Immediate next actions (in plan order)

1. Tell me what to do with the `data/passport_data` mystery folders (both copies) before anything touches them.
2. Decide how to bring the 734-image `~/Downloads/passport_data` set into `Donut_2.0/data/<card_type>/images/` — likely via `rename.py` (in-place on your original files — confirm before I run it) then a copy/move step.
3. Re-verify the 5 flagged rows above against their source images (or drop them).
4. Run `make_holdout.py data/real 10` — locks in the held-out test set before any augmentation touches this data again.
5. Rerun `add_tokens.py` so the processor vocab matches the current 106-type unified schema.
6. Run `vision_label.py` per card type once raw images are in place (needs your own `GOOGLE_API_KEY`), then verify in Label Studio, then `convert_labelstudio.py`.
7. `build_multi_type_dataset.py` → `sanity_check_dataset.py data/multi_type` → fix anything it flags.
8. Train on the A40 server (`train.py`), monitor via WandB, download the checkpoint back.
9. Run `batch_inference.py` against `data/holdout/*` and check field-level + MRZ-checksum accuracy.

`serve/` (FastAPI wrapper) is intentionally out of scope for this iteration.
