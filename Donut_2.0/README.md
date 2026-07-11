# Donut 2.0 — Passport / KYC Card Parser (model pipeline only)

Continuation of the `Donut` project on branch `donut-2.0`. Scope for this
iteration: data annotation → training data prep → training → POC validation.
**Serving (`serve/`, FastAPI) is explicitly deferred to a later iteration.**

See [STATUS.md](STATUS.md) for the current, honest state of the data/model
(not the aspirational one) and the immediate next actions.

## Setup

```bash
conda create -n donut2 python=3.12 -y
conda activate donut2
pip install -r requirements.txt
```

## Phase 1 — Data Annotation

Goal: clean, verified annotations for every card type, with a held-out test
set that never gets trained on.

1. Get a free Gemini API key at https://aistudio.google.com/app/apikey.
   **Never** put it in a file — export it in your terminal session only:
   ```bash
   export GOOGLE_API_KEY=AIza...
   ```
2. Drop raw images into `data/<card_type>/images/` (folders already created
   for `are_fed_card`, `cod`, `zwe`; `data/real` already has the Indian
   passport seed set).
3. Pre-annotate with Gemini:
   ```bash
   python src/vision_label.py data/are_fed_card/images data/are_fed_card/metadata_suggested.jsonl are_fed_card
   python src/vision_label.py data/cod/images           data/cod/metadata_suggested.jsonl           cod_passport
   python src/vision_label.py data/zwe/images           data/zwe/metadata_suggested.jsonl           zwe_passport
   ```
4. Import into Label Studio alongside the images, verify every field by
   hand — MRZ lines character-by-character, dates in `DD/MM/YYYY`, anything
   Gemini flagged with `_review_flag` (MRZ checksum failure) gets extra
   scrutiny.
5. Export from Label Studio, convert to training format:
   ```bash
   python src/convert_labelstudio.py data/are_fed_card/ls_export.json data/are_fed_card/metadata.jsonl are_fed_card
   ```
6. Carve out a held-out test set **before** touching augmentation — this
   only needs to be run once per card type, it's seeded and idempotent:
   ```bash
   python src/make_holdout.py data/real 10
   python src/make_holdout.py data/cod 10
   ```
   This moves the images out of `data/<type>/images` into
   `data/holdout/<card_type>/` and strips them from `metadata.jsonl` so
   they physically cannot leak into training.

## Phase 2 — Training Data Prep

Goal: turn verified samples into a large, augmented, correctly-tokenized
training set.

1. Augment each verified (post-holdout) set:
   ```bash
   python src/augment.py   # edit real_dir/output_dir/augmentations_per_image at the bottom for each card type
   ```
2. Spot-check ~20 random augmented images by eye — drop anything with an
   unrealistic transform.
3. Register tokens for the **current** `schemas.py` (rerun this any time a
   card type or field is added — it does not update itself):
   ```bash
   python src/add_tokens.py
   ```
4. Build the combined multi-type dataset once every type has a
   `metadata.jsonl`:
   ```bash
   python src/build_multi_type_dataset.py
   ```
5. Sanity-check before burning any GPU time — this also verifies every
   token used in the data actually exists in the processor vocab (catches
   a stale `add_tokens.py` run):
   ```bash
   python src/sanity_check_dataset.py data/multi_type
   ```

## Phase 3 — Training (A40 server)

1. Upload `Donut_2.0/` to the server under `/data`, not root (root disk
   fills up fast).
2. Confirm GPU is free: `nvidia-smi`.
3. `python src/train.py` — trains on `data/multi_type` with inverse-frequency
   class balancing across card types, logs to WandB, early-stops on
   `eval_loss` plateau.
4. Download the resulting checkpoint (`checkpoints/donut-multitype-final`)
   back to local.

## Phase 4 — POC Validation

1. Run inference on the held-out set:
   ```bash
   python src/batch_inference.py data/holdout/indian_passport batch_results_holdout indian_passport
   ```
2. Check field-level accuracy by hand against the held-out `metadata.jsonl`.
3. `mrz_validator.py` runs automatically inside `inference.py` — check the
   `mrz_validation` block in the output for checksum passes.
4. Note which fields fail most — that's the next thing to fix (more data
   for that field, a schema/prompt tweak, etc.), not a reason to move to
   serving yet.

## Phase 5 — Scale-up (parallel track)

Same Phase 1 pipeline, new `card_type` per document type. Add the schema to
`schemas.py`, rerun `add_tokens.py`, collect + label + convert + fold into
`build_multi_type_dataset.py`. Target 1K verified samples across types
before the next training run.

## Rules carried over from the plan

- Held-out test images: never augmented, never trained on.
- MRZ lines: always verify manually — never trust Gemini's output blindly.
- `data/` and `*.jsonl` are gitignored — nothing here gets pushed to GitHub.
- Commit code and configs after every working milestone.
- `serve/` is out of scope until the model itself is validated.
