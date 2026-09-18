"""
纯代码评分：不调 LLM。
- RAG 召回率（recall@1 / recall@3）
- 结构完整性（各节点是否产出非空结果）
"""

from langchain.messages import AIMessage
from tools.rag.retriever import retrieve_theories_raw
from tools.rag.fact_retriever import retrieve_facts_raw
from tools.memory import SqliteStore


# ─── RAG 召回率 ──────────────────────────────────────

def score_rag(scenario: dict) -> dict:
    """
    用 ground_truth.expected_theories 计算 recall@1 和 recall@3。

    Returns:
        {
            "retrieved": ["基本归因错误", ...],
            "relevant": ["基本归因错误"],
            "recall_1": 1 | 0,
            "recall_3": float,   # hits / len(relevant)
            "hits": ["基本归因错误"],
            "misses": []
        }
    """
    relevant = set(scenario.get("ground_truth", {}).get("relevant_theories", []))
    if not relevant:
        return {
            "retrieved": [], "relevant": [],
            "recall_1": None, "recall_3": None,
            "hits": [], "misses": [],
            "note": "ground_truth.relevant_theories 为空，跳过 RAG 评分"
        }

    claims = [{"content": scenario.get("description", "")}]
    cards = retrieve_theories_raw(claims, top_k=3)
    retrieved = [c["name"] for c in cards]

    hits = [name for name in retrieved if name in relevant] # 第一条理论命中了
    misses = [name for name in relevant if name not in retrieved]

    recall_1 = 1 if retrieved and retrieved[0] in relevant else 0
    recall_3 = len(hits) / len(relevant) if relevant else 0

    return {
        "retrieved": retrieved,
        "relevant": list(relevant),
        "recall_1": recall_1,
        "recall_3": round(recall_3, 2),
        "hits": hits,
        "misses": misses,
    }


# ─── 结构完整性 ───────────────────────────────────────

def score_structure(result: dict) -> dict:
    """
    检查 graph 输出结构是否完整。

    Returns:
        {
            "checks": {
                "graph_completed": true,
                "all_claims_have_evidence": true,
                "all_claims_have_alternatives": true,
                "info_symmetry_produced": false,
                "contradictions_produced": true
            },
            "passed": 4,
            "total": 5,
            "score": 0.8
        }
    """
    checks = {}

    # 1. graph 是否完成（有 messages 且最后一条非空）
    messages = result.get("messages", [])
    checks["graph_completed"] = any(isinstance(message, AIMessage) for message in messages)
    # 只有summary节点会往messages中塞AIMessage

    # 2. 每个 claim 是否有 evidence
    claims = result.get("all_claims", [])
    if claims:
        checks["all_claims_have_evidence"] = all(
            c.get("evidence_from_signals") for c in claims
        )
        checks["all_claims_have_alternatives"] = all(
            c.get("alternative_explanations") for c in claims
        )
    else:
        checks["all_claims_have_evidence"] = False
        checks["all_claims_have_alternatives"] = False

    # 3. info_symmetry 是否有产出
    info_sym = result.get("info_symmetry", {})
    checks["info_symmetry_produced"] = bool(info_sym) and any(
        v.get("is_sufficient") is not None for v in info_sym.values() # info_symmetry节点被执行就行了
        if isinstance(v, dict)
    )

    # 4. contradictions 是否有产出（至少有一个 contradiction）
    contradictions = result.get("contradictions", {})
    checks["contradictions_produced"] = bool(contradictions) and any(
        len(items) > 0 for items in contradictions.values()
        if isinstance(items, list)
    )

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "score": round(passed / total, 2) if total else 0,
    }


# ─── fact 检索召回率 ───────────────────────────────────

def score_fact_rag(case: dict) -> dict:
    """
    消费 fact_retrieval_golden.json 的一条 case，算 fact 检索 recall@3。

    跑法（M2 定测法）：seed_facts 写入独立内存 store → retrieve_facts_raw([query_claim], top_k=3)
    → 查 relevant_actions 是否进 top-3 → recall@k。

    BGE 不可用时返回 recall_3=None（跳过，不算失败）。
    """
    seed_facts = case.get("seed_facts", [])
    relevant = set(case.get("relevant_actions", []))
    query_claim = case.get("query_claim", "").strip()

    if not seed_facts or not query_claim:
        return {"recall_3": None, "note": "case 缺少 seed_facts/query_claim，跳过"}
    if not relevant:
        return {"recall_3": None, "note": "case 缺少 relevant_actions，跳过"}

    # 每个 case 独立内存 store，隔离（真实跑也是独立会话）
    store = SqliteStore(":memory:")
    for i, f in enumerate(seed_facts):
        store.put(("facts", "main"), f"seed_{i}", f)

    try:
        facts = retrieve_facts_raw([{"content": query_claim}], store, top_k=3)
    except Exception as e:
        return {"recall_3": None, "note": f"fact 检索失败（BGE 不可用？）：{e}"}

    retrieved_actions = [f.get("action", "") for f in facts]
    hits = [a for a in retrieved_actions if a in relevant]
    recall_3 = len(hits) / len(relevant) if relevant else 0

    return {
        "query_claim": query_claim,
        "retrieved": retrieved_actions,
        "relevant": list(relevant),
        "recall_3": round(recall_3, 2),
        "hits": hits,
        "misses": [a for a in relevant if a not in retrieved_actions],
    }
