"""Knowledge retriever combining vector search and keyword matching."""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass

from rag.embedding import DEFAULT_EMBEDDER, EmbeddingModel
from rag.loader import KnowledgeLoader
from rag.splitter import Chunk, MarkdownSplitter, SemanticSplitter, TextSplitter
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    category: str
    similarity: float


class KnowledgeRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: EmbeddingModel | None = None,
        splitter: TextSplitter | None = None,
        similarity_threshold: float = 0.1,  # 新增可配置阈值
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or DEFAULT_EMBEDDER
        self.similarity_threshold = similarity_threshold

        if splitter is None:
            try:
                self.splitter = SemanticSplitter(self.embedder)
                logger.info("使用语义分割器 (SemanticSplitter)")
            except Exception as e:
                warnings.warn(f"语义分割器初始化失败，回退到 MarkdownSplitter: {e}", stacklevel=2)
                self.splitter = MarkdownSplitter()
        else:
            self.splitter = splitter

    def build_index(self) -> None:
        documents = KnowledgeLoader.load_all()
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.splitter.split(doc))
        if all_chunks:
            texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedder.embed(texts)
            self.vector_store.add(all_chunks, embeddings)
            self.vector_store.save()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed([query])[0]
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            similarity_threshold=self.similarity_threshold,
        )
        retrieved: list[RetrievedChunk] = []
        for chunk, similarity in results:
            retrieved.append(
                RetrievedChunk(
                    content=chunk.content,
                    source=chunk.source,
                    category=chunk.category,
                    similarity=similarity,
                )
            )
        return retrieved

    def retrieve_with_keywords(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        vector_results = self.retrieve(query, top_k=top_k * 2)
        keywords = self._extract_keywords(query)
        scored: list[tuple[RetrievedChunk, float]] = []
        for result in vector_results:
            keyword_score = self._keyword_match_score(result.content, keywords)
            combined_score = result.similarity * 0.7 + keyword_score * 0.3
            scored.append((result, combined_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [result for result, _ in scored[:top_k]]

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        pattern = re.compile(r"[a-zA-Z0-9_一-龥]+")
        tokens = pattern.findall(text)
        return {token.lower() for token in tokens if len(token) >= 2}

    @staticmethod
    def _keyword_match_score(content: str, keywords: set[str]) -> float:
        if not keywords:
            return 0.0
        content_lower = content.lower()
        matched = sum(1 for keyword in keywords if keyword in content_lower)
        return matched / len(keywords)

    @classmethod
    def from_loaded_store(cls, similarity_threshold: float = 0.1) -> KnowledgeRetriever:
        vector_store = VectorStore.load()
        return cls(vector_store=vector_store, similarity_threshold=similarity_threshold)


# 默认实例（可使用环境变量覆盖阈值）
DEFAULT_RETRIEVER = KnowledgeRetriever.from_loaded_store()
