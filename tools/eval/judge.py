"""
LLM-as-Judge：用 json_llm 独立评估 joker 的分析质量。
两个维度：覆盖度（关键点都提了吗） + 正确性（方向对吗）。

设计原则：
- json_object 结构化输出 + extract_json 提取（与 5 个 graph 节点一致，不用 bind_tools）
- 多选题 → 代码查表算分（与 joker v1 checklist 模式一致）
- 对抗性 prompt——裁判默认姿态是"找问题"，不是"确认正确"
- 双跑取均值——降低单次评分的偶然性
"""

import asyncio

from langchain.messages import SystemMessage

from tools.llm.chat_llm import json_llm as llm
from tools.llm.safe_llm_call import safe_json_call_async


# ─── 格式化 joker 输出 ────────────────────────────────

def _format_output(result: dict) -> str:
    """把 graph 输出格式化为裁判可读的文本块。"""
    parts = []

    claims = result.get("all_claims", [])
    for i, c in enumerate(claims, 1):
        content = c.get("content", "?")
        direction = c.get("direction", "?")
        parts.append(f"### Claim {i}: {content} (direction={direction})")

        evidence = c.get("evidence_from_signals", {})
        if evidence:
            parts.append("**证据分析：**")
            for sig_key, ev in evidence.items():
                if isinstance(ev, dict):
                    parts.append(f"- {sig_key}: {ev.get('content', '?')} (credibility={ev.get('credit_score', '?')})")

        alts = c.get("alternative_explanations", {})
        if alts:
            parts.append("**替代解释：**")
            for sig_key, alt in alts.items():
                if isinstance(alt, dict):
                    parts.append(f"- {sig_key}: {alt.get('alternative', '?')} (你认为: {alt.get('youthink', '?')})")

    contradictions = result.get("contradictions", {})
    if contradictions:
        parts.append("## 矛盾记录")
        for claim_content, items in contradictions.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        parts.append(f"- {claim_content}: {item.get('reason', '?')}")

    info_sym = result.get("info_symmetry", {})
    if info_sym:
        parts.append("## 信息对称性")
        for key, val in info_sym.items():
            if isinstance(val, dict):
                parts.append(f"- {key}: user_knew={val.get('user_knew')}, ta_knew={val.get('ta_knew')}, sufficient={val.get('is_sufficient')}")

    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        parts.append(f"## 最终回复\n{last.content if hasattr(last, 'content') else last}")

    return "\n\n".join(parts) if parts else "（无输出）"


# ─── 覆盖度评分 ───────────────────────────────────────

def _to_int(v, default=0):
    """LLM 可能把数字返回成字符串（如 "3"），安全转 int。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


async def _run_single_coverage(prompt: str, llm) -> dict | None:
    """单次覆盖度评分调用。失败返回 None。"""
    args = await safe_json_call_async(
        llm,
        [SystemMessage(content=prompt)],
        node_name="COVERAGE_JUDGE"
    )
    if args is None:
        return None
    return {
        "covered_count": _to_int(args.get("covered_count", 0)),
        "missed_count": _to_int(args.get("missed_count", 0)),
        "comment": str(args.get("comment", "")),
    }


async def judge_coverage(scenario: dict, result: dict, llm=llm) -> dict:
    """
    LLM 判定：期望覆盖的关键方向是否在分析中出现。双跑取均值。
    """
    description = scenario.get("description", "")
    expected = scenario.get("ground_truth", {}).get("expected_direction", "")
    output = _format_output(result)

    if not expected.strip():
        return {
            "covered_count": 0, "missed_count": 0, "comment": "无期望方向，跳过覆盖度评分",
            "score": None, "raw_response": {}
        }

    prompt = f"""你是关系认知分析的评审专家。你的任务是找出分析中**缺失的关键方向**——你的默认姿态是挑毛病，不是确认正确。

**用户场景：**
{description}

**期望分析应覆盖的关键方向：**
{expected}

**joker 的实际分析输出：**
{output}

判断标准：
- 期望方向中的每个关键论述点，是否在 joker 分析中被明确提及或讨论了？
- "提及"不只是出现关键词——必须有一定的展开讨论
- 如果某个关键点完全没有被讨论，算 missed
- 如果被简要提及但没有展开，也算 missed（我们要的是"覆盖"，不是"擦边"）

请统计 covered（完全或充分覆盖）和 missed（缺失或擦边）的关键点数，并给出一个简短评论。

**输出要求：只输出一个 JSON 对象，不要任何其他文字或 markdown 代码块，格式：**
{{"covered_count": 3, "missed_count": 1, "comment": "一句话总结"}}"""

    # 双跑取均值（并发）
    runs = await asyncio.gather(
        _run_single_coverage(prompt, llm=llm),
        _run_single_coverage(prompt, llm=llm),
    )
    runs = [r for r in runs if r is not None]  # 过滤None

    if not runs:
        return {
            "covered_count": 0, "missed_count": 0, "comment": "LLM 调用失败（双跑均失败）",
            "score": None, "raw_response": {}
        }

    covered = round(sum(r["covered_count"] for r in runs) / len(runs))
    missed = round(sum(r["missed_count"] for r in runs) / len(runs))
    total = covered + missed

    return {
        "covered_count": covered,
        "missed_count": missed,
        "comment": runs[0]["comment"],  # 取第一跑的评语
        "score": round(covered / total, 2) if total > 0 else 0,
        "raw_response": runs[0],
    }


# ─── 正确性评分 ───────────────────────────────────────

# 多选题 → 分数映射
_DIRECTION_MAP = {"A": 1.0, "B": 0.6, "C": 0.3, "D": 0.0}
_THEORY_MAP    = {"A": 1.0, "B": 0.5, "C": 0.2, "D": 0.0}
_ACTION_MAP    = {"A": 1.0, "B": 0.5, "C": 0.0}


async def _run_single_correctness(prompt: str, llm) -> dict | None:
    """单次正确性评分调用。失败返回 None。"""
    args = await safe_json_call_async(
        llm,
        [SystemMessage(content=prompt)],
        node_name="CORRECTNESS_JUDGE"
    )
    if args is None:
        return None
    d = str(args.get("direction", "B")).upper()[0]
    t = str(args.get("theory_usage", "B")).upper()[0]
    a = str(args.get("actionable", "B")).upper()[0]
    return {
        "direction": d,
        "theory_usage": t,
        "actionable": a,
        "comment": args.get("comment", ""),
        "scores": {
            "direction": _DIRECTION_MAP.get(d, 0.5),
            "theory": _THEORY_MAP.get(t, 0.3),
            "actionable": _ACTION_MAP.get(a, 0.3),
        },
    }


async def judge_correctness(scenario: dict, result: dict, llm=llm) -> dict:
    """
    LLM 判定：分析的质量。三道单选题 → 查表算分。双跑取均值。
    """
    description = scenario.get("description", "")
    expected = scenario.get("ground_truth", {}).get("expected_direction", "")
    output = _format_output(result)

    prompt = f"""你是关系认知分析的评审专家。你的任务是**严格评估** joker 分析的质量。你的默认姿态是挑剔——只有真正达到标准才给高分。

**用户场景：**
{description}

**正确的分析方向（仅供参考——不要求逐字匹配）：**
{expected}

**joker 的实际分析输出：**
{output}

请回答以下三道单选题（每题选一个字母）：

1. **核心分析方向是否正确？**
   A) 正确——准确识别了认知根源或核心问题，分析方向与期望一致
   B) 部分正确——大致方向对，但深度不足或遗漏了重要维度
   C) 偏离——关注了次要问题，忽略了最核心的认知偏差/关系动态
   D) 错误——分析方向与正确方向相反，或完全曲解了场景

2. **理论运用是否恰当？（如果分析中没有提到任何心理学理论，选 D）**
   A) 恰当——理论被准确应用于场景，不是贴标签
   B) 可接受——理论相关，但使用表面化（提了名字但没展开）
   C) 牵强——硬套了一个不匹配的理论
   D) 缺失——该用理论的地方没有用，或分析中没有理论框架

3. **分析是否提供了可操作的洞察？**
   A) 有——给出了用户可以直接理解和应用的具体洞察
   B) 一般——给了方向但不具体（如"你要多沟通"但没有说怎么做）
   C) 无——分析停留在抽象描述层面，用户读完不知道怎么办

**输出要求：只输出一个 JSON 对象，不要任何其他文字或 markdown 代码块，格式：**
{{"direction": "A", "theory_usage": "B", "actionable": "B", "comment": "一句话总结"}}"""

    # 双跑取均值（并发）
    runs = await asyncio.gather(
        _run_single_correctness(prompt, llm=llm),
        _run_single_correctness(prompt, llm=llm),
    )
    runs = [r for r in runs if r is not None]

    if not runs:
        return {
            "direction": "?", "theory_usage": "?", "actionable": "?",
            "comment": "LLM 调用失败（双跑均失败）",
            "scores": {"direction": 0, "theory": 0, "actionable": 0},
            "total": 0,
            "raw_response": {},
        }

    # 平均分数
    avg_scores = {
        "direction": round(sum(r["scores"]["direction"] for r in runs) / len(runs), 2),
        "theory": round(sum(r["scores"]["theory"] for r in runs) / len(runs), 2),
        "actionable": round(sum(r["scores"]["actionable"] for r in runs) / len(runs), 2),
    }
    total = round(sum(avg_scores.values()) / 3, 2)

    return {
        "direction": runs[0]["direction"],
        "theory_usage": runs[0]["theory_usage"],
        "actionable": runs[0]["actionable"],
        "comment": runs[0]["comment"],
        "scores": avg_scores,
        "total": total,
        "raw_response": runs[0],
    }
