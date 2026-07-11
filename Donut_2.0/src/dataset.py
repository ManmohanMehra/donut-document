from torch.utils.data import Dataset
from transformers import DonutProcessor
from PIL import Image
import json, random
from pathlib import Path

class PassportDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        processor: DonutProcessor,
        split: str = "train",
        max_length: int = 512,
        val_split: float = 0.1
    ):
        self.processor = processor
        self.max_length = max_length
        self.data_dir = Path(data_dir)

        with open(self.data_dir / "metadata.jsonl") as f:
            records = [json.loads(line) for line in f]

        # Reproducible split
        random.seed(42)
        random.shuffle(records)
        split_idx = int(len(records) * (1 - val_split))
        self.records = records[:split_idx] if split == "train" else records[split_idx:]

        print(f"[{split}] {len(self.records)} samples loaded")

    def __len__(self):
        return len(self.records)

    def _gt_to_token_sequence(self, gt: dict) -> str:
        """
        {"card_type": "indian_passport", "surname": "SINGH", ...}
        \u2192
        <s_indian_passport><s_surname>SINGH</s_surname>...</s_indian_passport>
        """
        gt = gt.copy()
        card_type = gt.pop("card_type")
        seq = f"<s_{card_type}>"
        for key, value in gt.items():
            seq += f"<s_{key}>{value or ''}</s_{key}>"
        seq += f"</s_{card_type}>"
        return seq

    def __getitem__(self, idx):
        record = self.records[idx]

        # Load and process image
        img = Image.open(
            self.data_dir / "images" / record["file_name"]
        ).convert("RGB")
        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze()

        # Build and tokenize target sequence
        target_seq = self._gt_to_token_sequence(record["ground_truth"])
        labels = self.processor.tokenizer(
            target_seq,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Mask padding from loss computation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
