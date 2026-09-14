"""
共享 LLM 调用工具：重试 + 结构化日志 + 成本记账
所有 LLM 调用节点统一使用此函数，替代裸 llm.bind_tools().invoke()
"""
import time

from tools.logger import get_logger
from tools.llm.tracked_llm import _extract_usage
from tools.llm.cost_tracker import record
from tools.llm._extract_json import extract_json


def safe_llm_call(llm, tools: list, messages: list, node_name: str = "LLM", max_retries: int = 2):
    """调用 LLM（带 bind_tools），tool_calls 为空 → 重试，全失败 → 返回 None。

    Args:
        llm: ChatOpenAI 实例
        tools: bind_tools 的工具列表
        messages: SystemMessage/HumanMessage 列表
        node_name: 用于日志前缀，如 "EVIDENCE"
        max_retries: 最大重试次数（总共 max_retries+1 次尝试）

    Returns:
        成功 → AIMessage（response.tool_calls 非空）
        失败 → None
    """
    log = get_logger()

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            response = llm.bind_tools(tools).invoke(messages)
            usage = _extract_usage(response)
            latency_ms = round((time.perf_counter() - start) * 1000)

            # token 记账（无论 tool_calls 是否为空，token 都已烧）
            record(node_name, usage["input_tokens"], usage["output_tokens"], latency_ms, attempt)

            if response.tool_calls:
                log.info(node_name, "LLM 调用成功",
                         attempt=attempt, latency_ms=latency_ms, **usage)
                return response
            else:
                if attempt < max_retries:
                    log.warn(node_name, "tool_calls 为空，重试中",
                             attempt=f"{attempt+1}/{max_retries}")
                else:
                    log.warn(node_name, "所有重试均返回空 tool_calls",
                             total_attempts=max_retries + 1)

        except Exception as e:
            latency_ms = round((time.perf_counter() - start) * 1000)
            if attempt < max_retries:
                log.warn(node_name, f"调用失败，重试中",
                         attempt=f"{attempt+1}/{max_retries}", error=str(e), latency_ms=latency_ms)
            else:
                log.error(node_name, f"所有重试全部失败",
                          total_attempts=max_retries + 1, error=str(e), latency_ms=latency_ms)

    return None


async def safe_llm_call_async(llm, tools: list, messages: list, node_name: str = "LLM", max_retries: int = 2):
    """safe_llm_call 的异步版：ainvoke + await，供 asyncio.gather 并发调用。

    与同步版唯一区别：`llm.bind_tools(tools).invoke(messages)` → `await ...ainvoke(messages)`。
    重试、日志、token 记账逻辑完全一致。
    """
    log = get_logger()

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            response = await llm.bind_tools(tools).ainvoke(messages)
            usage = _extract_usage(response)
            latency_ms = round((time.perf_counter() - start) * 1000)

            record(node_name, usage["input_tokens"], usage["output_tokens"], latency_ms, attempt)

            if response.tool_calls:
                log.info(node_name, "LLM 调用成功",
                         attempt=attempt, latency_ms=latency_ms, **usage)
                return response
            else:
                if attempt < max_retries:
                    log.warn(node_name, "tool_calls 为空，重试中",
                             attempt=f"{attempt+1}/{max_retries}")
                else:
                    log.warn(node_name, "所有重试均返回空 tool_calls",
                             total_attempts=max_retries + 1)

        except Exception as e:
            latency_ms = round((time.perf_counter() - start) * 1000)
            if attempt < max_retries:
                log.warn(node_name, f"调用失败，重试中",
                         attempt=f"{attempt+1}/{max_retries}", error=str(e), latency_ms=latency_ms)
            else:
                log.error(node_name, f"所有重试全部失败",
                          total_attempts=max_retries + 1, error=str(e), latency_ms=latency_ms)

    return None


async def  safe_json_call_async(llm, messages: list, node_name: str = "LLM", max_retries: int = 2):
    """safe_llm_call_async 的 json_object 版：ainvoke + extract_json，供 json_llm 结构化输出。

    与 safe_llm_call_async 的区别：
    - 不 bind_tools（json_object 与 tools 互斥）
    - 成功判据从「tool_calls 非空」换成「extract_json 能解析出 dict」
    - 返回解析后的 dict（不是 AIMessage）；拿不到合法 JSON → 重试，全失败 → None
    """
    log = get_logger()

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            response = await llm.ainvoke(messages)
            usage = _extract_usage(response)
            latency_ms = round((time.perf_counter() - start) * 1000)

            # token 记账（无论 JSON 是否解析成功，token 都已烧）
            record(node_name, usage["input_tokens"], usage["output_tokens"], latency_ms, attempt)

            text = response.content if isinstance(response.content, str) else str(response.content)
            data = extract_json(text)
            if data is not None:
                log.info(node_name, "LLM 调用成功",
                         attempt=attempt, latency_ms=latency_ms, **usage)
                return data
            else:
                if attempt < max_retries:
                    log.warn(node_name, "JSON 解析失败，重试中",
                             attempt=f"{attempt+1}/{max_retries}")
                else:
                    log.warn(node_name, "所有重试均未返回合法 JSON",
                             total_attempts=max_retries + 1)

        except Exception as e:
            latency_ms = round((time.perf_counter() - start) * 1000)
            if attempt < max_retries:
                log.warn(node_name, f"调用失败，重试中",
                         attempt=f"{attempt+1}/{max_retries}", error=str(e), latency_ms=latency_ms)
            else:
                log.error(node_name, f"所有重试全部失败",
                          total_attempts=max_retries + 1, error=str(e), latency_ms=latency_ms)

    return None
