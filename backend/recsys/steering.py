"""Natural-language steering with the FashionCLIP text encoder.

FashionCLIP embeds text and images in the same space, so a phrase like "in red" or
"summer dresses" can be compared with every product photo directly. The steer is used two ways:
  - retrieval: FAISS search with  normalise(mean(liked images) + gamma * text)  ("like these, but red"),
    or the text alone before any likes;
  - ranking: z-scored text->image similarity is added to each candidate's score.
"""
from __future__ import annotations

import re

import numpy as np

FILLER = re.compile(
    r"\b(show me|i want|i'd like|i would like|give me|more like (this|these)|something|but|please|maybe|some)\b",
    re.IGNORECASE)


def clean_query(text: str) -> str:
    text = FILLER.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ,.") or text.strip()


def prompt(text: str) -> str:
    return f"a photo of {clean_query(text)} clothing"


class FashionClipText:
    """Text tower only (the image tower isn't needed at serving time)."""

    def __init__(self, hf_id="patrickjohncyh/fashion-clip"):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        torch.set_grad_enabled(False)
        model = CLIPModel.from_pretrained(hf_id)
        self.text_model, self.projection = model.text_model.eval(), model.text_projection.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(hf_id)
        self._torch = torch
        del model.vision_model

    def encode(self, text: str) -> np.ndarray:
        tok = self.tokenizer([prompt(text)], padding=True, truncation=True, max_length=77, return_tensors="pt")
        with self._torch.inference_mode():
            pooled = self.text_model(**tok).pooler_output
            v = self.projection(pooled)[0].numpy().astype(np.float32)
        return v / np.linalg.norm(v)


class HashingText:
    """Deterministic stand-in for tests (no model download)."""

    def __init__(self, dim=512):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(clean_query(text).lower())) % 2**32)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)
