"""Knowledge QA skill using RAG retrieval with optimized LLM prompt."""

from __future__ import annotations

from rag.retriever import KnowledgeRetriever, RetrievedChunk
from rag.splitter import Chunk

from .llm_client import LLMClient
from .skill_base import KnowledgeSkill, SkillResult


class KnowledgeQASkill(KnowledgeSkill):
    """知识问答技能：RAG 检索 + LLM 答案合成"""

    def __init__(
        self,
        retriever: KnowledgeRetriever | None = None,
        llm: LLMClient | None = None,
        temperature: float = 0.3,
    ) -> None:
        super().__init__(llm=llm, temperature=temperature)
        self._retriever = retriever
        self._retriever_loaded = False

    def _get_retriever(self) -> KnowledgeRetriever:
        if not self._retriever_loaded:
            if self._retriever is None:
                from rag.retriever import DEFAULT_RETRIEVER

                self._retriever = DEFAULT_RETRIEVER
            self._retriever_loaded = True
        return self._retriever

    def name(self) -> str:
        return "knowledge_qa"

    def description(self) -> str:
        return (
            "Answer thermodynamics knowledge questions using retrieved documents "
            "from the knowledge base with LLM-synthesized responses."
        )

    def _is_parameter_query(self, query: str) -> bool:
        param_keywords = [
            "参数",
            "parameter",
            "临界",
            "critical",
            "acentric",
            "偏心因子",
            "Tc",
            "Pc",
            "常数",
            "constant",
            "系数",
            "coefficient",
            "压力范围",
            "pressure",
            "温度范围",
            "temperature",
        ]
        return any(kw in query.lower() for kw in param_keywords)

    def _is_feature_query(self, query: str) -> bool:
        feature_keywords = ["特点", "特征", "特性", "描述", "介绍", "是什么", "什么是", "define", "meaning"]
        return any(kw in query.lower() for kw in feature_keywords)

    def _is_concept_query(self, query: str) -> bool:
        concept_keywords = ["概念", "concept", "定义"]
        return any(kw in query.lower() for kw in concept_keywords)

    def _chunk_to_retrieved(self, chunk: Chunk, similarity: float = 0.1) -> RetrievedChunk:
        return RetrievedChunk(
            content=chunk.content,
            source=chunk.source,
            category=chunk.category,
            similarity=similarity,
        )

    def execute(self, query: str, **kwargs) -> SkillResult:
        try:
            retriever = self._get_retriever()
            top_k = 5 if (self._is_parameter_query(query) or self._is_feature_query(query)) else 3
            chunks: list[RetrievedChunk] = retriever.retrieve_with_keywords(query, top_k=top_k)

            # 如果是参数查询或特征查询，额外加载 fundamentals 文档
            if self._is_parameter_query(query) or self._is_feature_query(query):
                from rag.embedding import DEFAULT_EMBEDDER
                from rag.loader import KnowledgeLoader
                from rag.splitter import SemanticSplitter

                all_docs = KnowledgeLoader.load_all()
                fundamental_docs = [d for d in all_docs if "fundamentals" in d.source]
                if fundamental_docs:
                    splitter = SemanticSplitter(DEFAULT_EMBEDDER)
                    additional_chunks: list[Chunk] = []
                    for doc in fundamental_docs:
                        additional_chunks.extend(splitter.split(doc))
                    if additional_chunks:
                        query_lower = query.lower()
                        for c in additional_chunks:
                            # 匹配相关性：查询中的关键词在 chunk 中出现
                            if any(kw in c.content.lower() for kw in query_lower.split() if len(kw) > 2):
                                chunks.append(self._chunk_to_retrieved(c, similarity=0.05))
        except Exception as e:
            return SkillResult(
                answer=f"❌ 知识库检索失败: {e}",
                sources=[],
                confidence=0.0,
            )

        if not chunks:
            return SkillResult(
                answer="📭 知识库中未找到相关信息。",
                sources=[],
                confidence=0.0,
            )

        context_parts = []
        sources = []
        total_similarity = 0.0
        for chunk in chunks:
            source_label = chunk.source.replace("\\", "/")
            context_parts.append(f"[来源: {source_label}]\n{chunk.content}")
            sources.append(chunk.source)
            total_similarity += chunk.similarity

        combined_context = "\n\n---\n\n".join(context_parts)

        base_system_prompt = """你是 ThermoEqui-Agent 的热力学知识助手，专注于化工热力学领域的准确知识问答。

## 任务
1. 仔细阅读用户问题和下方知识库上下文
2. 提取与问题直接相关的核心事实，不要添加上下文之外的信息
3. 用清晰的逻辑组织回答，优先给出最直接的答案

## 输出格式
- 核心结论（直接回答用户问题，1-2句话）
- 补充细节（支撑依据，用编号列出）
- 引用来源（在相关句末标注 [来源: 文件名]）

## 约束
- 回答长度控制在 150-300 字
- 如果上下文充分：直接回答，要点清晰
- 如果上下文部分相关：明确指出「根据现有信息，可以推断...」
- 如果上下文不相关或不足：明确回复「⚠️ 当前知识库未包含该问题的完整信息」
- 不添加上下文之外的任何知识
- 不重复用户的问题
"""

        if self._is_parameter_query(query):
            extra = """
## 特别注意
用户询问的是**参数信息**。请从上下文中提取具体的**参数名称**和**数值/范围**，例如：
- 偏心因子 (acentric factor)
- 临界温度 (critical temperature)
- 临界压力 (critical pressure)
- 二元交互参数 (binary interaction parameter)
请尽可能列出所有提到的参数，即使没有数值也要列出名称。
"""
        elif self._is_feature_query(query):
            extra = """
## 特别注意
用户询问的是模型的**特点/特征**。请从上下文中提取关键特征，如：
- 理论基础（如局部组成理论、状态方程等）
- 适用体系（极性/非极性、烃类等）
- 优点和局限性
- 与其它模型的区别
"""
        elif self._is_concept_query(query):
            extra = """
## 特别注意
用户询问的是**概念定义**。请给出清晰的定义，然后列出关键特征或组成部分。
"""
        else:
            extra = ""

        system_prompt = base_system_prompt + extra

        answer = self._synthesize_with_llm(
            query=query,
            context=combined_context,
            system_prompt=system_prompt,
            temperature=self._temperature,
        )

        return SkillResult(
            answer=answer,
            sources=list(set(sources)),
            confidence=total_similarity / len(chunks) if chunks else 0.0,
            metadata={
                "chunk_count": len(chunks),
                "llm_used": self._llm_available,
                "temperature": self._temperature,
                "query_type": (
                    "parameter"
                    if self._is_parameter_query(query)
                    else "feature"
                    if self._is_feature_query(query)
                    else "concept"
                    if self._is_concept_query(query)
                    else "general"
                ),
            },
        )

    def supports_intent(self, intent: str) -> bool:
        return intent in {"CONCEPT_QA", "KNOWLEDGE_QA", "MODEL_SELECTION_QA"}
