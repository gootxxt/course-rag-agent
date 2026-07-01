from __future__ import annotations

import hashlib
import math

import numpy as np

from .config import Settings
from .utils import tokenize


class EmbeddingModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.using_fallback = False
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
        except Exception as exc:
            if not settings.allow_hash_embedding_fallback:
                raise RuntimeError(
                    f"Failed to load embedding model {settings.embedding_model!r}. "
                    "Install sentence-transformers/model files or enable fallback."
                ) from exc
            self.using_fallback = True

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.model is not None:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        return [self._hash_embedding(text) for text in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        dim = self.settings.hash_embedding_dim
        vec = np.zeros(dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(float(np.dot(vec, vec)))
        if norm > 0:
            vec /= norm
        return vec.tolist()

