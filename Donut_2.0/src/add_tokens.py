from transformers import DonutProcessor
import json
import sys
sys.path.append('src')
from schemas import get_all_tokens

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Pulls every token for every card_type currently defined in schemas.py.
# Rerun this (and retrain) any time a card type is added to SCHEMAS —
# the saved processor's vocab does not update itself.
new_tokens = get_all_tokens()

processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
processor.save_pretrained("checkpoints/donut-passport-processor")

print(f"Added {len(new_tokens)} tokens.")
print(f"New vocab size: {len(processor.tokenizer)}")
