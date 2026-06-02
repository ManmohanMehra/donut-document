from transformers import DonutProcessor
import json
import sys
sys.path.append('src')
from schemas import get_all_tokens

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Automatically pull the exact 34 tokens we defined in schemas.py
new_tokens = get_all_tokens()

processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
processor.save_pretrained("checkpoints/donut-passport-processor")

print(f"Added {len(new_tokens)} tokens.")
print(f"New vocab size: {len(processor.tokenizer)}")
