"""
记忆 fact 检索：从 store 的 ("facts","main") namespace 读 fact，用 BGE 语义检索 top-K 相关 fact。

复用 embedder 的 BGE model（与理论卡检索共用同一个实例）。query 由 pending claims 拼接。
fact 的检索文本 = action + signal_type + source_ref（行为的语义内容）。
"""
import numpy as np

from tools.rag.embedder import embed, embed_query


def _build_fact_text(fact: dict) -> str:
    """拼接 fact 用于 embedding 的检索文本。"""
    return " ".join([
        fact.get("action", ""),
        fact.get("signal_type", ""),
        fact.get("source_ref", ""),
    ])


def _load_facts(store) -> list[tuple[str, dict]]:
    """从 store 加载所有 fact，返回 [(fact_id, fact), ...]。"""
    if store is None:
        return []
    try:
        items = store.search(("facts", "main"))
    except Exception:
        return []
    facts = []
    for it in items:
        if isinstance(it.value, dict):
            facts.append((it.key, it.value))
    return facts


def retrieve_facts_raw(claims: list[dict], store, top_k: int = 3) -> list[dict]:
    """检索 top_k 相关 fact，返回原始 fact dict 列表（未格式化）。"""
    if not claims or store is None:
        return []
    query = " ".join([c.get("content", "") for c in claims if c.get("content", "")])
    if not query.strip():
        return []

    facts = _load_facts(store)
    if not facts:
        return []

    fact_ids = [fid for fid, _ in facts]
    fact_texts = [_build_fact_text(f) for _, f in facts]

    fact_embs = embed(fact_texts)          # (n, dim), L2 归一化（passage 侧不加 instruction）
    query_emb = embed_query(query)         # (1, dim)（query 侧加 instruction）
    scores = np.dot(fact_embs, query_emb.T).flatten()  # (n,)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [facts[i][1] for i in top_indices]


def retrieve_facts(claims: list[dict], store, top_k: int = 3) -> str:
    """检索 top_k 相关 fact，返回格式化的 prompt 文本。无结果时返回空字符串。"""
    facts = retrieve_facts_raw(claims, store, top_k=top_k)
    if not facts:
        return ""
    return _format_facts(facts)


def _format_facts(facts: list[dict]) -> str:
    """将 fact 列表格式化为 prompt 可注入的文本块。"""
    lines = [
        "## 历史行为事实（来自长期记忆）",
        "以下是从长期记忆检索到的、与当前主张相关的历史行为事实，分析时可作为补充证据：",
    ]
    for i, fact in enumerate(facts, 1):
        person = "用户" if fact.get("person") == "user" else "对方"
        action = fact.get("action", "?")
        confidence = fact.get("confidence", 0.5)
        source = fact.get("source_ref", "")
        line = f"{i}. [{person}] {action}（置信度 {confidence}）"
        if source:
            line += f" 原文：{source}"
        lines.append(line)
    return "\n".join(lines)
