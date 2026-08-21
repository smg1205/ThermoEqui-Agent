"""Embedding utilities for text encoding."""

from __future__ import annotations

import hashlib
import os
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
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install with: pip install sentence-transformers"
                ) from exc
        return self._model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [np.array(e) for e in embeddings]


def _create_default_embedder() -> EmbeddingModel:
    """Create the best available embedder.

    Prefers SentenceTransformer for real semantic embeddings.
    Falls back to HashEmbedding when the model or dependencies are unavailable
    (e.g. offline environments, CI without model weights).

    Set ``HF_ENDPOINT`` to a mirror (e.g. ``https://hf-mirror.com``) if
    huggingface.co is unreachable.
    """
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        embedder = SentenceTransformerEmbedding()
        # Eagerly load to fail fast if the model cannot be downloaded/loaded
        embedder._get_model()
        return embedder
    except Exception:
        warnings.warn(
            "SentenceTransformer unavailable — falling back to HashEmbedding. "
            "Install sentence-transformers and ensure model access for semantic retrieval.",
            stacklevel=2,
        )
        return HashEmbedding()


DEFAULT_EMBEDDER = _create_default_embedder()
