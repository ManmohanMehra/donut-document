import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from dataset import PassportDataset

# \u2500\u2500 Load processor and model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
processor = DonutProcessor.from_pretrained("checkpoints/donut-passport-processor")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Expand decoder embeddings to match new vocab size
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Decoder config \u2014 set start token to your primary card type
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
    ["<s_indian_passport>"]
)[0]
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# \u2500\u2500 Datasets \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
train_dataset = PassportDataset("data/augmented", processor, split="train")
val_dataset   = PassportDataset("data/augmented", processor, split="val")

training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/checkpoints/donut-passport-finetuned",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,    # Ensure eval doesn't OOM
    gradient_accumulation_steps=4,
    fp16=True,
    eval_strategy="epoch",           # Evaluate after each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,    # Use the best weights found
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    push_to_hub=False
)


# \u2500\u2500 Train \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model("checkpoints/donut-passport-final")
processor.save_pretrained("checkpoints/donut-passport-final")
print("Training complete. Model saved to checkpoints/donut-passport-final")
