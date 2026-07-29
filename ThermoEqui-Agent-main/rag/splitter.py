"""Text splitting utilities for knowledge documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    category: str
    chunk_id: int
    total_chunks: int


class TextSplitter(Protocol):
    def split(self, document: Document) -> list[Chunk]: ...


class MarkdownSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        content = document.content
        chunks: list[Chunk] = []
        if document.metadata and document.metadata.get("file_type") == "markdown":
            sections = self._split_by_headings(content)
            for section in sections:
                chunks.extend(self._split_text(section, document))
        else:
            chunks.extend(self._split_text(content, document))
        return chunks

    def _split_by_headings(self, content: str) -> list[str]:
        pattern = r"(#{1,3}\s+.+?$)"
        sections = re.split(pattern, content, flags=re.MULTILINE)
        result: list[str] = []
        current = ""
        for section in sections:
            if re.match(pattern, section):
                if current:
                    result.append(current.strip())
                current = section
            else:
                current += section
        if current:
            result.append(current.strip())
        return [s for s in result if s]

    def _split_text(self, text: str, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        sentences = self._split_sentences(text)
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(
                    Chunk(
                        content=" ".join(current_chunk),
                        source=document.source,
                        category=document.category,
                        chunk_id=len(chunks),
                        total_chunks=0,
                    )
                )
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(
                Chunk(
                    content=" ".join(current_chunk),
                    source=document.source,
                    category=document.category,
                    chunk_id=len(chunks),
                    total_chunks=0,
                )
            )

        for i, chunk in enumerate(chunks):
            chunks[i] = chunk.__class__(
                **{**chunk.__dict__, "total_chunks": len(chunks)}
            )

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]


class SemanticSplitter:
    def __init__(
        self,
        embedder,
        max_chunk_size: int = 500,
        min_chunk_size: int = 50,
        similarity_threshold: float = 0.5,
    ) -> None:
        self.embedder = embedder
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold

    def split(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    content=sentences[0],
                    source=document.source,
                    category=document.category,
                    chunk_id=0,
                    total_chunks=1,
                )
            ]

        embeddings = self.embedder.embed(sentences)

        similarities = []
        for i in range(len(embeddings) - 1):
            v1 = embeddings[i]
            v2 = embeddings[i + 1]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = float(np.dot(v1, v2) / (norm1 * norm2))
            similarities.append(sim)

        # 初步切分：根据相似度阈值和最大长度
        split_indices = [0]
        current_start = 0
        accumulated_len = 0

        for i, sim in enumerate(similarities):
            accumulated_len += len(sentences[i])

            if sim < self.similarity_threshold:
                if accumulated_len >= self.min_chunk_size:
                    split_indices.append(i + 1)
                    current_start = i + 1
                    accumulated_len = 0
            elif accumulated_len > self.max_chunk_size:
                split_indices.append(i + 1)
                current_start = i + 1
                accumulated_len = 0

        if split_indices[-1] < len(sentences):
            split_indices.append(len(sentences))

        raw_chunks = []
        for j in range(len(split_indices) - 1):
            start = split_indices[j]
            end = split_indices[j + 1]
            chunk_sentences = sentences[start:end]
            if chunk_sentences:
                raw_chunks.append(" ".join(chunk_sentences))

        # 修复小块合并逻辑：使用贪心合并，保证每个块至少 min_chunk_size
        merged = []
        buffer = ""
        for chunk in raw_chunks:
            if not buffer:
                buffer = chunk
            elif len(buffer) + len(chunk) <= self.max_chunk_size:
                buffer += " " + chunk
            else:
                merged.append(buffer)
                buffer = chunk
        if buffer:
            merged.append(buffer)

        # 处理最后一块可能过小的情况（如果只有一块，直接保留）
        if len(merged) > 1 and len(merged[-1]) < self.min_chunk_size:
            # 将最后一块合并到前一块（如果合并后不超过 max）
            if len(merged[-2]) + len(merged[-1]) <= self.max_chunk_size:
                merged[-2] += " " + merged[-1]
                merged.pop()
            else:
                # 如果合并会超长，则独立保留（但记录警告）
                pass

        result = []
        for idx, text in enumerate(merged):
            if not text.strip():
                continue
            result.append(
                Chunk(
                    content=text.strip(),
                    source=document.source,
                    category=document.category,
                    chunk_id=idx,
                    total_chunks=len(merged),
                )
            )
        return result

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]