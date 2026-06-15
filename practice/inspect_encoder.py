from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base"
)

# Print encoder architecture summary
print(model.encoder)

# Count encoder parameters
enc_params = sum(p.numel() for p in model.encoder.parameters())
print(f"Encoder params: {enc_params / 1e6:.1f}M")

# Trace output shape through encoder
dummy = torch.randn(1, 3, 960, 1280)
with torch.no_grad():
    enc_out = model.encoder(pixel_values=dummy)
print(f"Encoder output shape: {enc_out.last_hidden_state.shape}")
# Expected: torch.Size([1, 588, 1024])
