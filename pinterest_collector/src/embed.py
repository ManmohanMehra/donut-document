"""
embed.py — DINOv2 image embeddings for visual similarity search.

DINOv2 is used instead of CLIP because we care about fine-grained visual
similarity between card layouts/textures/colours, not semantic/text
alignment — CLIP tends to conflate "these are both blue rectangular things"
with "these are both ID cards."

Usage as a library:
    from embed import Embedder, cosine_similarity
    embedder = Embedder()
    vec = embedder.embed_image("path/to/card.jpg")   # L2-normalized np.ndarray
"""
import numpy as np
from PIL import Image

DEFAULT_MODEL = "facebook/dinov2-base"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        # CLS token — DINOv2's standard image-level embedding
        vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, image_paths: list[str]) -> np.ndarray:
        return np.stack([self.embed_image(p) for p in image_paths])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Both vectors must already be L2-normalized (Embedder.embed_image does this)."""
    return float(np.dot(a, b))


def best_match(query: np.ndarray, index_vectors: np.ndarray) -> tuple[int, float]:
    """Return (index, similarity) of the closest vector in index_vectors to query."""
    sims = index_vectors @ query
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])
