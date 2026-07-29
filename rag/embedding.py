"""Embedding utilities for text encoding."""

from __future__ import annotations

import hashlib
import warnings
from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[np.ndarray]: ...


class HashEmbedding:
    """Deterministic hash-based embedding for offline testing."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for text in texts:
            embedding = np.zeros(self.dimension, dtype=np.float32)
            hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            for i in range(self.dimension):
                byte_index = i % len(hash_bytes)
                embedding[i] = (hash_bytes[byte_index] / 255.0) * 2 - 1
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return embeddings


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [np.array(e) for e in embeddings]


try:
    DEFAULT_EMBEDDER = HashEmbedding()
except ImportError:
    DEFAULT_EMBEDDER = HashEmbedding()
