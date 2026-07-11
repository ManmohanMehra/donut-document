import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from dataset import PassportDataset

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_HEIGHT   = 1920
IMAGE_WIDTH    = 1280
NUM_EPOCHS     = 40
DATA_DIR       = "data/multi_type"   # combined & interleaved metadata.jsonl for all shortlisted types
OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "donut-multitype-finetuned")
PROCESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "donut-passport-processor")

MAX_OVERSAMPLE = 3.0  # cap: smallest type never gets more than 3x the weight of the largest

# ── Load processor and model ───────────────────────────────────────────────────
processor = DonutProcessor.from_pretrained(PROCESSOR_PATH, local_files_only=True)
processor.image_processor.size = {"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH}
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

model.decoder.resize_token_embeddings(len(processor.tokenizer))

# decoder_start_token_id is now a generic pad token so the model can predict
# the correct <s_card_type> token from the image (per-example start, not fixed).
# During inference, decoder_input_ids is always set explicitly per card type.
model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
model.config.pad_token_id  = processor.tokenizer.pad_token_id
model.config.eos_token_id  = processor.tokenizer.eos_token_id

# ── Datasets ──────────────────────────────────────────────────────────────────
train_dataset = PassportDataset(DATA_DIR, processor, split="train")
val_dataset   = PassportDataset(DATA_DIR, processor, split="val")


# ── Weighted sampler — inverse-frequency, capped at MAX_OVERSAMPLE ─────────────
class MultiTypeTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with inverse-frequency WeightedRandomSampler for class balance."""

    def _get_train_dataloader(self) -> DataLoader:
        card_types = [r["ground_truth"]["card_type"] for r in self.train_dataset.records]
        counts = Counter(card_types)
        max_count = max(counts.values())
        weights = [
            min(max_count / counts[ct], MAX_OVERSAMPLE) for ct in card_types
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=10,
    report_to="tensorboard",
    run_name="donut-multitype-v2-hires",
    push_to_hub=False,
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = MultiTypeTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

print(f"Starting training | Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH} | Epochs: {NUM_EPOCHS}")
trainer.train()

final_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "donut-multitype-final")
trainer.save_model(final_dir)
processor.save_pretrained(final_dir)
print(f"Training complete. Model saved to {final_dir}")
