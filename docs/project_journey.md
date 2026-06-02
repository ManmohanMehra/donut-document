# 🗺️ Project Journey: Donut Passport Parser

This document tracks the technical evolution of the Indian Passport parser, from initial concept to high-accuracy model training.

## Phase 1: Foundation & Ground Truth
- **Objective**: Define what the model should extract.
- **Actions**: 
  - Analyzed Indian Passport layouts.
  - Created a precise JSON schema (`passport_number`, `dob`, `sex`, etc.).
  - Annotated a small "seed" dataset of real images using Label Studio standards.
  - Formatted data into the `metadata.jsonl` format required by Donut.

## Phase 2: Vocabulary Expansion (Smart Embeddings)
- **Objective**: Teach the model the "language" of passports.
- **Actions**:
  - Implemented `add_tokens.py`.
  - Added custom XML-style tags: `<s_indian_passport>`, `<s_dob>`, etc.
  - Resized model embeddings using **Multivariate Normal Distribution** initialization (mean-resizing) to ensure the new tokens inherited knowledge from existing pre-trained weights.

## Phase 3: Data Augmentation (Scaling from 10 to 700)
- **Objective**: Create enough data for deep learning without manual annotation.
- **Actions**:
  - Used `Albumentations` to apply realistic distortions:
    - Geometric transforms (rotations, perspective shifts).
    - Lighting effects (brightness, contrast, shadows).
    - Compression artifacts (simulating mobile phone photos).
  - Scaled the dataset to **700+ high-quality samples**.

## Phase 4: Serving & Validation Pipeline
- **Objective**: Build the "Production Wrapper" before the final training.
- **Actions**:
  - Developed `src/inference.py` with multi-step output cleaning.
  - Developed `src/mrz_validator.py` to check the **ICAO TD3 checksums** (mathematical proof of data accuracy).
  - Built a **FastAPI** server in `serve/app.py` for real-time processing.

## Phase 5: Cloud GPU Fine-Tuning (Current)
- **Objective**: Train the model for 15-30 epochs on an NVIDIA T4.
- **Actions**:
  - Optimized the training script for Kaggle environments.
  - Applied **Gradient Checkpointing** and **Image Downsampling (1280x960)** to fit the model into 15GB VRAM.
  - Implemented **Early Stopping** to prevent overfitting and capture the best version of the weights.

## Next Steps
- [ ] Download final weights from Kaggle.
- [ ] Replace local weights in `checkpoints/donut-passport-final`.
- [ ] Final end-to-end latency testing.
- [ ] Docker containerization for deployment.
