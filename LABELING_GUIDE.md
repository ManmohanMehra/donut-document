# Pre-Labeling Guide — Iteration 2 (ARE-FED-CARD, COD, ZWE)

Quick reference for running Gemini Vision pre-labeling on the 3 new card types,
then reviewing in Label Studio before training.

---

## 1. Pull the Repo on Work Laptop

```bash
git clone https://github.com/<your-username>/<your-repo>.git Donut
cd Donut
```

Or if already cloned:
```bash
git pull origin main
```

---

## 2. Set Up the Conda Environment

```bash
conda create -n donut python=3.12 -y
conda activate donut
pip install -r requirements.txt
pip install google-generativeai pillow tqdm
```

---

## 3. Get a Free Gemini API Key

1. Go to: **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key (starts with `AIza...`)

> **NEVER paste the key into any file in this repo.**
> Set it only as an environment variable in your terminal session.

---

## 4. Set the API Key (Terminal Only — Never in a File)

**Mac / Linux:**
```bash
export GOOGLE_API_KEY=AIza...your-key-here...
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY = "AIza...your-key-here..."
```

This key lives only in your terminal session. It disappears when you close the terminal and is never written to disk.

---

## 5. Add Your Images

Place raw images in the correct folders (create them if they don't exist):

```
data/
  are_fed_card/
    images/       ← UAE Federal ID Card images here (47 images)
  cod/
    images/       ← DRC Passport images here (45 images)
  zwe/
    images/       ← Zimbabwe Passport images here (60 images)
```

> Note: `data/` is in `.gitignore` — images will NOT be pushed to GitHub.

---

## 6. Run Pre-Labeling (Gemini Vision)

Run from the repo root. Each command processes one card type and writes suggested labels.

```bash
cd Donut   # make sure you're in the repo root

python src/vision_label.py data/are_fed_card/images  data/are_fed_card/metadata_suggested.jsonl  are_fed_card

python src/vision_label.py data/cod/images           data/cod/metadata_suggested.jsonl           cod_passport

python src/vision_label.py data/zwe/images           data/zwe/metadata_suggested.jsonl           zwe_passport
```

**Free tier rate**: 15 requests/minute (~4s per image).
Total time for 152 images: ~11 minutes. Leave it running.

The script will print:
- A progress bar per card type
- Any MRZ checksum failures (flagged for review)
- Final count of labeled vs. errored images

Output files (also gitignored):
- `data/are_fed_card/metadata_suggested.jsonl`
- `data/cod/metadata_suggested.jsonl`
- `data/zwe/metadata_suggested.jsonl`

---

## 7. Review in Label Studio

1. Open Label Studio and create a project for each card type
2. Import images + upload `metadata_suggested.jsonl` as pre-annotations
3. Review each image — correct wrong fields, fill in nulls, fix MRZ lines
4. Pay extra attention to images with `_review_flag` (MRZ checksum failed)
5. Export annotations as **JSON** format

---

## 8. Convert Label Studio Export to Training Format

```bash
python src/convert_labelstudio.py data/are_fed_card/ls_export.json  data/are_fed_card/metadata.jsonl  are_fed_card

python src/convert_labelstudio.py data/cod/ls_export.json           data/cod/metadata.jsonl           cod_passport

python src/convert_labelstudio.py data/zwe/ls_export.json           data/zwe/metadata.jsonl           zwe_passport
```

---

## 9. Build the Combined Training Dataset

Run only after all 4 card types have a `metadata.jsonl` (IND is already done):

```bash
python src/build_multi_type_dataset.py
```

Output: `data/multi_type/metadata.jsonl` + `data/multi_type/images/`

---

## Safety Checklist Before Any `git push`

- [ ] `GOOGLE_API_KEY` is NOT in any `.py`, `.md`, `.json`, or `.env` file
- [ ] `git status` shows no `.env` files, no `credentials.json`
- [ ] `git diff --staged` has no API keys or secrets in it
- [ ] `data/` folder contents are NOT staged (it's gitignored)

```bash
# Quick check — should return nothing sensitive
git diff --staged | grep -i "AIza\|api_key\|secret\|password"
```

---

## Card Type Reference

| Card Type      | Folder            | # Images | `card_type` arg  |
|----------------|-------------------|----------|------------------|
| UAE Federal ID | `data/are_fed_card` | 47     | `are_fed_card`   |
| DRC Passport   | `data/cod`          | 45     | `cod_passport`   |
| Zimbabwe Pass. | `data/zwe`          | 60     | `zwe_passport`   |
| Indian Pass.   | `data/real`         | 111    | already labeled  |
