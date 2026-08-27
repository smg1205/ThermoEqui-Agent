"""SQLite-based conversation memory with hybrid vector + keyword retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.embedding import DEFAULT_EMBEDDER, EmbeddingModel
from schemas.domain import Intent

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "conversation_memory.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    intent      TEXT NOT NULL,
    components  TEXT,
    task_summary TEXT,
    embedding   BLOB,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_session ON conversation_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_intent ON conversation_memory(intent);
"""


@dataclass(frozen=True)
class MemoryRecord:
    """A single conversation memory entry."""

    question: str
    answer: str
    intent: str
    components: list[str]
    task_summary: str | None
    similarity: float = 0.0


class ConversationMemory:
    """SQLite store with hybrid vector + keyword retrieval for same-session memory."""

    def __init__(
        self,
        db_path: Path | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.db_path = db_path or DB_PATH
        self.embedder = embedder or DEFAULT_EMBEDDER
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.executescript(_CREATE_TABLE)
            conn.close()

    def save(
        self,
        session_id: str,
        question: str,
        answer: str,
        intent: Intent | str,
        components: list[str] | None = None,
        task_summary: str | None = None,
    ) -> None:
        """Save a Q&A pair with embedding to SQLite."""
        intent_str = intent.value if isinstance(intent, Intent) else str(intent)
        components_json = json.dumps(components or [], ensure_ascii=False)
        embedding = self.embedder.embed([question])[0]
        embedding_blob = embedding.astype(np.float32).tobytes()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """INSERT INTO conversation_memory
                   (session_id, question, answer, intent, components, task_summary, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, question, answer, intent_str, components_json, task_summary, embedding_blob),
            )
            conn.commit()
            conn.close()

    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.1,
    ) -> list[MemoryRecord]:
        """Hybrid retrieval: vector similarity (0.7) + keyword match (0.3)."""
        query_embedding = self.embedder.embed([query])[0]
        keywords = self._extract_keywords(query)

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            rows = conn.execute(
                """SELECT question, answer, intent, components, task_summary, embedding
                   FROM conversation_memory WHERE session_id = ?
                   ORDER BY created_at DESC LIMIT 50""",
                (session_id,),
            ).fetchall()
            conn.close()

        if not rows:
            return []

        scored: list[tuple[float, MemoryRecord]] = []
        for question, answer, intent, components_json, task_summary, embedding_blob in rows:
            stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            vector_sim = self._cosine_similarity(query_embedding, stored_embedding)
            keyword_score = self._keyword_match_score(question + " " + answer, keywords)
            combined = vector_sim * 0.4 + keyword_score * 0.6
            if combined > similarity_threshold:
                components_list = json.loads(components_json) if components_json else []
                record = MemoryRecord(
                    question=question,
                    answer=answer,
                    intent=intent,
                    components=components_list,
                    task_summary=task_summary,
                    similarity=combined,
                )
                scored.append((combined, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    _CJK_STOPWORDS = frozenset(
        "的了吗呢吧啊呀哦是在有和与及或但而把这被让给向从对为以于"
        "比好坏多少大小上下中内外左右前后还又不也没已再就只还"
        "什么怎么为什么哪里如何是否可以应该需要能能够会要想"
        "问答说看做找求计算结概述介绍一下这种那些那个这个"
        "体系系统条件下常压"
    )

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract keywords with weighted importance.

        Returns (ascii_keywords, cjk_keywords) separately for weighted scoring.
        ASCII terms (model names, component IDs) are high-value;
        CJK single chars are kept only if not common stop words.
        """
        keywords: set[str] = set()
        # Extract ASCII tokens (model names, component names, etc.)
        ascii_tokens = re.findall(r"[a-zA-Z0-9_-]+", text)
        for token in ascii_tokens:
            if len(token) >= 2:
                keywords.add(token.lower())
        # Extract multi-char CJK words
        cjk_words = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        for word in cjk_words:
            keywords.add(word)
        # Extract single CJK chars, excluding stop words
        for char in text:
            if "\u4e00" <= char <= "\u9fff" and char not in ConversationMemory._CJK_STOPWORDS:
                keywords.add(char)
        return keywords

    @staticmethod
    def _keyword_match_score(content: str, keywords: set[str]) -> float:
        if not keywords:
            return 0.0
        content_lower = content.lower()
        # Weight: ASCII keywords (>=2 chars, contains latin) get 2x weight
        total_weight = 0.0
        matched_weight = 0.0
        for kw in keywords:
            is_ascii = any(c.isascii() and c.isalpha() for c in kw)
            weight = 2.0 if is_ascii else 1.0
            total_weight += weight
            if kw.lower() in content_lower:
                matched_weight += weight
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("DELETE FROM conversation_memory WHERE session_id = ?", (session_id,))
            conn.commit()
            conn.close()
