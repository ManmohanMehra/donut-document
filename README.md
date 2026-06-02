# 🍩 Donut Indian Passport Parser

A state-of-the-art document understanding system based on the **Donut (Document Understanding Transformer)** architecture. This project fine-tunes a vision-encoder-decoder model to extract structured JSON data from Indian Passport images without using OCR.

## 🚀 Project Overview
This system replaces expensive, generic OCR/LLM pipelines with a specialized, local model that:
1. **Reads images directly** (End-to-end Vision).
2. **Extracts critical fields**: Passport Number, DOB, Sex, Place of Birth, and Expiry.
3. **Validates Integrity**: Integrated MRZ (Machine Readable Zone) checksum validation.
4. **Serves via API**: Fast inference using FastAPI.

## 🏗️ Architecture
- **Model**: `naver-clova-ix/donut-base` (Fine-tuned).
- **Backend**: Python (PyTorch, Transformers).
- **API**: FastAPI with Uvicorn.
- **Serving**: Dockerized for easy deployment.

## 📁 Project Structure
- `src/`: Core logic (Training, Inference, Dataset loader).
- `data/`: Raw and augmented training data.
- `checkpoints/`: Local storage for the fine-tuned model weights.
- `serve/`: FastAPI application code.
- `docs/`: Technical guides and roadmap.

## 🛠️ Performance Optimizations
- **Gradient Checkpointing**: Enabled for low VRAM training.
- **Image Downsampling**: Optimized at 1280x960 for speed and accuracy.
- **Mixed Precision (FP16)**: For faster GPU calculations.

---
*Created with ❤️ for Advanced Document AI.*
