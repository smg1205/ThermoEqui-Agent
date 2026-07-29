"""Retrieval-Augmented Generation module for knowledge-grounded responses."""

import logging

from rag.loader import Document, KnowledgeLoader
from rag.retriever import KnowledgeRetriever
from rag.vector_store import VectorStore

__all__ = ["Document", "KnowledgeLoader", "KnowledgeRetriever", "VectorStore"]

# 配置基本日志（可选）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rag():
    """简易测试：加载、分块、建索引、检索，并诊断语义分割效果。"""
    print("=" * 50)
    print("开始 RAG 功能测试...")

    docs = KnowledgeLoader.load_all()
    print(f"✅ 加载到 {len(docs)} 个文档")
    if not docs:
        print("❌ 未找到任何文档，请确保 knowledge/ 目录下有 .md 或 .yaml 文件。")
        return

    for doc in docs[:3]:
        print(f"   - {doc.title or '无标题'} (来源: {doc.source}, 分类: {doc.category})")

    retriever = KnowledgeRetriever()

    print("\n⏳ 正在构建索引...")
    retriever.build_index()
    print(f"✅ 索引构建完成，向量库共 {retriever.vector_store.size} 个向量块")

    print("\n" + "=" * 50)
    print("📖 语义分割效果诊断（仅展示前 3 个文档的块结构）")
    print("=" * 50)

    for idx, doc in enumerate(docs[:3]):
        chunks = retriever.splitter.split(doc)
        print(f"\n📄 文档 {idx+1}: {doc.source} (共 {len(chunks)} 个语义块)")
        if len(chunks) == 0:
            print("   ⚠️ 该文档分割后没有任何块（可能内容为空）")
            continue
        for i, chunk in enumerate(chunks):
            preview = chunk.content[:80].replace("\n", " ") + ("..." if len(chunk.content) > 80 else "")
            print(f"  块 {i+1}: [{len(chunk.content)} 字符] {preview}")

    test_query = "热力学平衡"
    print(f"\n🔍 检索测试：查询 '{test_query}'")
    results = retriever.retrieve(test_query, top_k=3)

    if results:
        print(f"✅ 检索到 {len(results)} 个结果:")
        for i, chunk in enumerate(results, 1):
            print(f"  {i}. 相似度 {chunk.similarity:.4f} | {chunk.source} [{chunk.category}]")
            print(f"     内容预览: {chunk.content[:80]}...")
    else:
        print("⚠️ 未检索到任何结果（可能相似度低于阈值 0.1，或知识库内容与查询无关）")

    print("\n🔍 关键词增强检索测试：")
    kw_results = retriever.retrieve_with_keywords(test_query, top_k=3)
    if kw_results:
        print(f"✅ 增强检索结果 ({len(kw_results)} 条):")
        for i, chunk in enumerate(kw_results, 1):
            print(f"  {i}. 相似度 {chunk.similarity:.4f} | {chunk.source}")
    else:
        print("⚠️ 无结果")

    print("\n🎉 测试完成！")


if __name__ == "__main__":
    test_rag()