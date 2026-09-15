"""
JOKER 后端 API（FastAPI + SSE 流式）。

用 LangGraph 的 astream_events 拿两类事件，推成 SSE：
1. 节点开始/完成（on_chain_start / on_chain_end）→ 进度推送
2. summary 节点的 LLM token（on_chat_model_stream）→ 逐字输出

运行：
    cd D:/final_joker
    python -m uvicorn achievement_graph.thought_v1.api:app --host 127.0.0.1 --port 8000
"""
import json
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.messages import HumanMessage
from pydantic import BaseModel

from .graph import app as graph_app
from tools.logger import set_run_id
from tools.llm.cost_report import log_run_summary


class StreamRequest(BaseModel):
    user_msg: str
    thread_id: str = ""

app = FastAPI(title="JOKER API")

# React 前端（Vite，localhost:5173）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 节点 → 前端展示的中文标签
NODE_LABELS = {
    "preprocess_node": "理解输入",
    "signals_node": "分析行为信号",
    "supervisor_node": "规划分析路径",
    "compression_node": "压缩历史上下文",
    "evidence_node": "收集证据",
    "alternative_explanation_node": "生成替代解释",
    "information_symmetry_node": "检查信息对称性",
    "contradictory_registration_node": "汇总矛盾",
    "summary_node": "生成诊断",
}

_PROGRESS_NODES = set(NODE_LABELS.keys())


def _sse(data: dict) -> str:
    """把 dict 编码成一条 SSE 事件（data: {json}\n\n）。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/stream")
async def stream(req: StreamRequest):
    """
    流式分析：节点进度 + summary 逐字。

    前端用 fetch + ReadableStream 读（不是 EventSource，因为要 POST 传长文本）。
    """
    user_msg = req.user_msg
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def gen():
        run_id = f"{thread_id}:{uuid.uuid4().hex[:8]}"  # 每轮随机，thread_id 前缀溯源；与 thread_id 解耦
        set_run_id(run_id)  # 本轮所有节点日志带上这个 run_id，串成一条调用链
        in_summary = False
        try:
            async for event in graph_app.astream_events(
                {"messages": [HumanMessage(content=user_msg)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                name = event.get("name", "")

                if kind == "on_chain_start" and name in _PROGRESS_NODES:
                    if name == "summary_node":
                        in_summary = True
                    yield _sse({"type": "node_start", "node": name, "label": NODE_LABELS[name]})

                elif kind == "on_chain_end" and name in _PROGRESS_NODES:
                    if name == "summary_node":
                        in_summary = False
                    yield _sse({"type": "node_end", "node": name})

                elif kind == "on_chat_model_stream" and in_summary:
                    chunk = event["data"]["chunk"]
                    token = chunk.content
                    if token:
                        yield _sse({"type": "token", "content": token})

            log_run_summary()  # 本轮 LLM 汇总（token/重试/延迟），进日志不发给用户
            yield _sse({"type": "done", "thread_id": thread_id})
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx 反代时关缓冲
        },
    )
