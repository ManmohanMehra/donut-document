# Donut 2.0 — Status Tracker

Honest numbers as of 2026-07-11, carried over from the `main` branch `Donut/`
project. This replaces the aspirational tracker in `next_steps_action_plan.md`
(which listed "150 passport samples collected — Done"; actual verified count
below is 19).

| Task | Status | Notes |
|---|---|---|
| Indian passport samples verified (`data/real`) | 19 verified, 1 orphan image | `pass_21.jpeg` has no `metadata.jsonl` entry — needs labeling or removal |
| Held-out test set (10 images, never trained on) | Not carved out yet | Run `make_holdout.py` before any augmentation |
| are_fed_card / cod / zwe raw images | Not collected | Folders created empty at `data/{are_fed_card,cod,zwe}/images/` |
| Gemini pre-annotation pipeline | Code ready (`vision_label.py`), migrated off deprecated SDK | Needs `GOOGLE_API_KEY` — never commit it, terminal env var only. Was on `google-generativeai` (end-of-support, no more fixes) — switched to `google-genai` |
| Label Studio verification | Not started for iteration 2 types | IND set (19) was verified manually in iteration 1 |
| Augmentation pipeline | Code ready (`augment.py`) | Last run produced 598 images in `data/augmented` from the pre-holdout 19 |
| Token registration | **Stale — must rerun** | `checkpoints/donut-passport-processor` has 19 tokens (iteration-1 schema only: indian_passport + foreign_passport). `schemas.py` now needs **44** tokens (5 card types). Run `add_tokens.py` before training on multi-type data. |
| Dataset sanity check | Script added (`sanity_check_dataset.py`) | Run after add_tokens.py + build_multi_type_dataset.py, before every training run |
| First Donut training run (A40) | Prior run exists (`checkpoints/donut-passport-final`, `donut-passport-finetuned/checkpoint-616`) — trained before token/schema expansion, indian_passport only | Retrain needed once multi-type data + refreshed tokens are ready |
| POC inference + validation | Code ready (`inference.py`, `batch_inference.py`, `mrz_validator.py`) | Blocked on a checkpoint trained with the current tokenizer |
| Broader multi-type dataset (`data/multi_type`) | Not built yet | `build_multi_type_dataset.py` will refuse to run until `are_fed_card`, `cod`, `zwe` each have a `metadata.jsonl` |
| Known data-quality issues to re-verify in Label Studio | Flagged, not fixed | See below — do not silently "fix" these without checking the source image |

## Known bad rows in `data/real/metadata.jsonl` (carried over as-is, not edited)

Per the plan's own rule ("MRZ lines — always verify manually, never trust
blindly"), these were not hand-fixed since the source images weren't
re-checked. Flag them for re-verification:

- `pass_2.png` — `date_of_expiry: "23705/2031"` (malformed, extra digit)
- `pass_12.jpg` — `passport_number: "Ԣ5101066"` (corrupted non-ASCII char, should likely be `T`); `mrz_line1` has a stray `C` before `HARWANT`; `mrz_line2` ends in `O` where a digit is expected
- `pass_10.jpg` (S1157960 record, `pass_22.jpg`) — `mrz_line2` ends in lowercase `o` where a digit is expected
- `pass_21.jpeg` — image present, no metadata record at all

## Immediate next actions (in plan order)

1. Re-verify the 5 flagged rows above against their source images (or drop them).
2. Run `make_holdout.py data/real 10` — locks in the held-out test set before any augmentation touches this data again.
3. Rerun `add_tokens.py` so the processor vocab matches the current 5-card-type schema.
4. Collect raw images into `data/are_fed_card/images`, `data/cod/images`, `data/zwe/images` (target counts from LABELING_GUIDE.md: 47 / 45 / 60).
5. Run `vision_label.py` on each (needs your own `GOOGLE_API_KEY`), then verify in Label Studio, then `convert_labelstudio.py`.
6. `build_multi_type_dataset.py` → `sanity_check_dataset.py data/multi_type` → fix anything it flags.
7. Train on the A40 server (`train.py`), monitor via WandB, download the checkpoint back.
8. Run `batch_inference.py` against `data/holdout/*` and check field-level + MRZ-checksum accuracy.

`serve/` (FastAPI wrapper) is intentionally out of scope for this iteration.
