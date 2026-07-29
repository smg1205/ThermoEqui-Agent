"""Vector store implementation using numpy for embedding storage and similarity search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rag.splitter import Chunk

EMBEDDING_DIMENSION = 384
VECTOR_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "vector_store.npz"


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk: Chunk
    embedding: np.ndarray


@dataclass
class VectorStore:
    records: list[EmbeddingRecord] = field(default_factory=list)

    def add(self, chunks: list[Chunk], embeddings: list[np.ndarray]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self.records.append(EmbeddingRecord(chunk=chunk, embedding=embedding))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.1,
    ) -> list[tuple[Chunk, float]]:
        if not self.records:
            return []
        embedding_matrix = np.array([record.embedding for record in self.records])
        similarities = self._cosine_similarity(query_embedding, embedding_matrix)
        indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in indices:
            if similarities[idx] > similarity_threshold:
                results.append((self.records[idx].chunk, float(similarities[idx])))
        return results

    @staticmethod
    def _cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        safe_norms = np.where(matrix_norms == 0, 1e-10, matrix_norms)
        similarities = np.dot(matrix, vector) / (safe_norms * vector_norm)
        return similarities

    def save(self, path: Path | None = None) -> None:
        save_path = path or VECTOR_STORE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embeddings": np.array([record.embedding for record in self.records]),
            "sources": np.array([record.chunk.source for record in self.records]),
            "categories": np.array([record.chunk.category for record in self.records]),
            "contents": np.array([record.chunk.content for record in self.records]),
            "chunk_ids": np.array([record.chunk.chunk_id for record in self.records]),
            "total_chunks": np.array([record.chunk.total_chunks for record in self.records]),
        }
        np.savez(save_path, **data)

    @classmethod
    def load(cls, path: Path | None = None) -> VectorStore:
        load_path = path or VECTOR_STORE_PATH
        if not load_path.exists():
            return cls()
        data = np.load(load_path, allow_pickle=True)
        store = cls()
        for i in range(len(data["sources"])):
            chunk = Chunk(
                content=str(data["contents"][i]),
                source=str(data["sources"][i]),
                category=str(data["categories"][i]),
                chunk_id=int(data["chunk_ids"][i]),
                total_chunks=int(data["total_chunks"][i]),
            )
            store.records.append(EmbeddingRecord(chunk=chunk, embedding=data["embeddings"][i]))
        return store

    @property
    def size(self) -> int:
        return len(self.records)