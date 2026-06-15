import os
import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from dataset import PassportDataset

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_HEIGHT   = 1920   # Increased from 1280 for better MRZ detail
IMAGE_WIDTH    = 1280   # Increased from 960
NUM_EPOCHS     = 40     # 30-40 is the target; EarlyStopping will stop earlier
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "donut-passport-finetuned")
PROCESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "donut-passport-processor")

# ── Load processor and model ───────────────────────────────────────────────────
processor = DonutProcessor.from_pretrained(PROCESSOR_PATH, local_files_only=True)
processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Expand decoder embeddings to match new vocab size
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Decoder config — set start token to your primary card type
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
    ["<s_indian_passport>"]
)[0]
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# ── Datasets ──────────────────────────────────────────────────────────────────
train_dataset = PassportDataset("data/augmented", processor, split="train")
val_dataset   = PassportDataset("data/augmented", processor, split="val")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,    # Saves VRAM for high resolution
    fp16=torch.cuda.is_available(), # FP16 only on CUDA
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=10,
    report_to="tensorboard",         # LOCAL logs only (Private & Safe)
    run_name="donut-passport-v2-hires",
    push_to_hub=False
)


# ── Train ─────────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

print(f"🚀 Starting training | Resolution: {IMAGE_HEIGHT}×{IMAGE_WIDTH} | Epochs: {NUM_EPOCHS}")
trainer.train()

final_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "donut-passport-final")
trainer.save_model(final_dir)
processor.save_pretrained(final_dir)
print(f"✅ Training complete. Model saved to {final_dir}")
