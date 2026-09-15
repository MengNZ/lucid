# -*- coding: utf-8 -*-
"""
评估 Runner：加载 golden dataset → 每个场景跑 joker graph → 四项评分 → 出报告。

运行方式：
    cd D:/my_joker
    python -m tools.eval.runner

要求：
    - golden_dataset.json 至少有两个完整标注的场景
    - DEEPSEEK_API_KEY 环境变量已设置
"""

import asyncio
import json
import os
import sys
import time
import uuid

# 确保项目根目录在 path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from achievement_graph.thought_v1.graph import graph
from tools.memory import SqliteStore
from tools.eval.scorer import score_rag, score_structure, score_fact_rag
from tools.eval.judge import judge_coverage, judge_correctness
from tools.logger import set_run_id
from tools.llm.cost_tracker import get_usage
from tools.llm.cost_report import usage_to_dict


# 并发跑 scenario 的上限（DeepSeek 限流 + 本机资源）。改成 1 等价于串行。
EVAL_CONCURRENCY = 3


# ─── 加载数据集 ───────────────────────────────────────

def _load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    valid = []
    for s in data:
        desc = s.get("description", "").strip()
        gt = s.get("ground_truth", {})
        has_direction = gt.get("expected_direction", "").strip()
        if desc and has_direction:
            valid.append(s)
        else:
            name = s.get("name") or s.get("id")
            print(f"  [skip] {name} (not annotated)")
    return valid


def _load_fact_cases() -> list[dict]:
    """加载 fact 检索测试集（M2 定测法，M3 跑数字）。"""
    path = os.path.join(os.path.dirname(__file__), "fact_retrieval_golden.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


# ─── 跑一个场景 ───────────────────────────────────────

async def _run_scenario(scenario: dict, thread_id: str) -> dict:
    """跑一个场景的 graph（异步），返回 result。

    thread_id 由调用方（run_eval）生成，作为 MemorySaver/store 的隔离键。
    每个 scenario 用独立 store（内存库），隔离 facts/profiles 跨场景污染。
    记账（set_run_id / get_usage）不在这里做，由 run_eval 统一开合——
    这样 graph + 两个 judge 的 LLM 都记进同一个 scenario 的账本。
    """
    description = scenario["description"]
    config = {"configurable": {"thread_id": thread_id}}

    store = SqliteStore(":memory:")
    app = graph.compile(checkpointer=MemorySaver(), store=store)

    return await app.ainvoke(
        {"messages": [HumanMessage(content=description)]},
        config=config,
    )


async def _run_one_scenario(sc: dict, sem: asyncio.Semaphore, total: int, idx: int) -> dict:
    """一个 scenario 的完整评估：RAG + graph + structure + coverage + correctness。

    独立 task 内 set_run_id —— contextvar 是 task 局部的，gather 时各场景互不污染。
    返回该 scenario 的结果 dict（含 cost），供 run_eval 汇总。
    """
    name = sc.get("name") or sc["id"]
    diff = sc.get("difficulty", "?")
    thread_id = f"eval-{sc['id']}-{uuid.uuid4().hex[:6]}"   # MemorySaver/store 隔离键
    run_id = f"eval-{sc['id']}-{uuid.uuid4().hex[:8]}"      # 本轮运行键：日志 + 账本（随机，与 thread_id 解耦）

    async with sem:                     # 限流：最多 sem 个场景同时跑
        set_run_id(run_id)              # 开账本：graph + 两个 judge 的 LLM 都记这本
        try:
            print(f"\n{'─' * 40}")
            print(f"[{idx}/{total}] {name} (difficulty={diff})")

            # 2a. RAG
            t0 = time.time()
            rag = score_rag(sc)
            print(f"  [RAG] recall@1={rag['recall_1']} recall@3={rag['recall_3']} "
                  f"retrieved={rag['retrieved']} ({time.time()-t0:.1f}s)")

            # 2b. Graph
            t0 = time.time()
            try:
                result = await _run_scenario(sc, thread_id)
                graph_ok = True
            except Exception as e:
                print(f"  [Graph] FAIL: {e}")
                result = {"messages": [], "all_claims": [], "info_symmetry": {}, "contradictions": {}}
                graph_ok = False
            graph_time = time.time() - t0
            print(f"  [Graph] {'OK' if graph_ok else 'FAIL'} ({graph_time:.1f}s)")

            # 2c. Structure
            structural = score_structure(result)
            print(f"  [Struct] {structural['passed']}/{structural['total']} "
                  f"({'OK' if structural['passed'] == structural['total'] else 'WARN'})")

            # 2d. Coverage (LLM)
            t0 = time.time()
            coverage = await judge_coverage(sc, result)
            print(f"  [Coverage] {coverage['score']} ({coverage.get('comment', '?')[:60]}) "
                  f"({time.time()-t0:.1f}s)")

            # 2e. Correctness (LLM)
            t0 = time.time()
            correctness = await judge_correctness(sc, result)
            print(f"  [Correctness] {correctness['total']} "
                  f"(dir={correctness['direction']} theory={correctness['theory_usage']} "
                  f"action={correctness['actionable']}) ({time.time()-t0:.1f}s)")

            usage = get_usage()         # 关账前读出总数
            return {
                "id": sc["id"],
                "name": name,
                "difficulty": diff,
                "scores": {
                    "rag": rag,
                    "structural": structural,
                    "coverage": coverage,
                    "correctness": correctness,
                },
                "cost": usage_to_dict(usage) if usage and usage.nodes else None,
            }
        finally:
            set_run_id(None)            # 关账本，防跨 scenario 累加


# ─── 汇总 ─────────────────────────────────────────────

def _summarize(scenario_results: list[dict]) -> dict:
    rag1_vals = []
    rag3_vals = []
    structural_vals = []
    coverage_vals = []
    correctness_vals = []

    for r in scenario_results:
        s = r["scores"]

        if s["rag"]["recall_1"] is not None:
            rag1_vals.append(s["rag"]["recall_1"])
        if s["rag"]["recall_3"] is not None:
            rag3_vals.append(s["rag"]["recall_3"])

        structural_vals.append(s["structural"]["score"])

        if s["coverage"]["score"] is not None:
            coverage_vals.append(s["coverage"]["score"])

        correctness_vals.append(s["correctness"]["total"])

    def _avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else 0

    rag1_avg = _avg(rag1_vals)
    rag3_avg = _avg(rag3_vals)
    struct_avg = _avg(structural_vals)
    coverage_avg = _avg(coverage_vals)
    correctness_avg = _avg(correctness_vals)

    weighted = round(
        rag3_avg * 0.25 + struct_avg * 0.20 + coverage_avg * 0.25 + correctness_avg * 0.30,
        2
    ) if (rag3_vals or coverage_vals) else 0

    if not rag3_vals and not coverage_vals:
        weighted = round(struct_avg * 0.40 + correctness_avg * 0.60, 2)

    return {
        "rag_recall_1_avg": rag1_avg,
        "rag_recall_3_avg": rag3_avg,
        "structural_avg": struct_avg,
        "coverage_avg": coverage_avg,
        "correctness_avg": correctness_avg,
        "weighted_total": weighted,
        "scenario_count": len(scenario_results),
    }


def _summarize_fact_rag(fact_results: list[dict]) -> dict:
    """fact 检索 recall 汇总。"""
    vals = [r.get("recall_3") for r in fact_results if r.get("recall_3") is not None]
    return {
        "recall_3_avg": round(sum(vals) / len(vals), 2) if vals else None,
        "case_count": len(fact_results),
        "evaluated": len(vals),
    }


def _summarize_cost(scenario_results: list[dict]) -> dict:
    """聚合所有 scenario 的成本（total 行求和）。"""
    keys = [
        "input_tokens", "output_tokens", "total_tokens",
        "calls", "retries", "retry_tokens",
        "latency_ms", "retry_latency_ms", "missing_usage",
    ]
    totals = {k: 0 for k in keys}
    for r in scenario_results:
        c = r.get("cost")
        if not c:
            continue
        t = c["total"]
        for k in keys:
            totals[k] += t.get(k, 0)
    totals["retry_token_ratio"] = (
        round(totals["retry_tokens"] / totals["total_tokens"], 2)
        if totals["total_tokens"] > 0 else None
    )
    return totals


# ─── 主入口 ───────────────────────────────────────────

async def run_eval(dataset_path: str | None = None):
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

    print("=" * 60)
    print("Joker v2 Phase 3 — Eval Runner")
    print("=" * 60)

    # 1. 加载
    print(f"\n[load] {dataset_path}")
    scenarios = _load_dataset(dataset_path)
    if not scenarios:
        print("[FAIL] No valid scenarios (need description + expected_direction)")
        return None
    print(f"  valid scenarios: {len(scenarios)}")

    # 2. 并发评估（每个 scenario 一个 task，独立 store + 独立 run_id）
    total_start = time.time()
    sem = asyncio.Semaphore(EVAL_CONCURRENCY)
    scenario_results = await asyncio.gather(
        *[_run_one_scenario(sc, sem, len(scenarios), i)
          for i, sc in enumerate(scenarios, 1)]
    )

    total_time = time.time() - total_start

    # 2.5 fact 检索 recall（独立测试集，M2 定测法）
    print(f"\n{'─' * 40}")
    print("Fact Retrieval Recall（fact_retrieval_golden.json）")
    fact_results = []
    for fc in _load_fact_cases():
        fr = score_fact_rag(fc)
        fact_results.append({"id": fc.get("id", "?"), **fr})
        print(f"  [{fc.get('id', '?')}] recall@3={fr.get('recall_3')} "
              f"retrieved={fr.get('retrieved', [])}")
    fact_summary = _summarize_fact_rag(fact_results)

    # 3. 汇总报告
    summary = _summarize(scenario_results)
    cost_summary = _summarize_cost(scenario_results)

    print(f"\n{'=' * 60}")
    print("Eval Report")
    print(f"{'=' * 60}")
    print(f"Scenarios: {summary['scenario_count']}")
    print(f"Total time: {total_time:.0f}s (~{total_time/summary['scenario_count']:.0f}s/scenario)")
    print()
    print(f"  RAG recall@1:      {summary['rag_recall_1_avg']:.2f}")
    print(f"  RAG recall@3:      {summary['rag_recall_3_avg']:.2f}")
    if fact_summary['recall_3_avg'] is not None:
        print(f"  Fact recall@3:     {fact_summary['recall_3_avg']:.2f}")
    print(f"  Structural:        {summary['structural_avg']:.2f}")
    print(f"  Coverage:          {summary['coverage_avg']:.2f}")
    print(f"  Correctness:       {summary['correctness_avg']:.2f}")
    print(f"  ─────────────────────────")
    print(f"  Weighted Total:    {summary['weighted_total']:.2f}")
    print()
    print(f"  Cost: in={cost_summary['input_tokens']} out={cost_summary['output_tokens']} "
          f"total={cost_summary['total_tokens']} tokens")
    print(f"        retries={cost_summary['retries']} retry_tokens={cost_summary['retry_tokens']} "
          f"retry_ratio={cost_summary['retry_token_ratio']}")
    print(f"        latency={cost_summary['latency_ms']/1000:.1f}s "
          f"retry_latency={cost_summary['retry_latency_ms']/1000:.1f}s")

    # 4. 按难度分组
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in scenario_results if r["difficulty"] == diff]
        if diff_results:
            diff_avg = sum(
                r["scores"]["correctness"]["total"] for r in diff_results
            ) / len(diff_results)
            print(f"  {diff}: {diff_avg:.2f} (n={len(diff_results)})")

    # 5. 写入 JSON 报告
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    report = {
        "eval_date": time.strftime("%Y-%m-%d %H:%M"),
        "summary": summary,
        "fact_rag": {"summary": fact_summary, "cases": fact_results},
        "cost": cost_summary,
        "scenarios": scenario_results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] {report_path}")

    return report


if __name__ == "__main__":
    asyncio.run(run_eval())
