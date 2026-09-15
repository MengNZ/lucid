# Lucid — 读不懂的情感

> 基于 LangGraph 的多节点认知分析管线，帮你发现恋爱中「以为自己看清了，其实没看清」的偏差。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2%2B-green)](https://github.com/langchain-ai/langgraph)
[![React](https://img.shields.io/badge/React-19%2B-61dafb)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**线上 Demo：** [https://bot.hzd-ms.com/joker-demo](https://bot.hzd-ms.com/joker-demo)

---

## 为什么做这个

高中三年，我给喜欢的女孩写了两年信。她只回过一封，短短几行。过年约她出来走，她说家里走不开。我想：哦，这就是拒绝。我报了北京的学校，她留在了合肥。

上个月才知道——她把每一封信按日期排好，放在一个铁盒子里。她说：「我不是不喜欢你，我只是太不会表达了。」

我花了三年「确认」她不喜欢我，然后用这个结论选了北京的学校。如果当时多问一句呢？

**人用感受推演对方的想法，感受是偏差工厂。** 我没有任何客观证据，却用三年建了一座自以为坚固的逻辑大厦——然后它在一个铁盒子面前塌了。

Lucid 想做的是：把「猜对方在想什么」这件事，从直觉的黑箱里拿出来，变成可检查、可反证、可校准的结构化流程。

## 这是什么？

Lucid 是一个**关系认知分析引擎**。你粘贴聊天记录或描述一段关系困惑，它会：

1. **提取你所有的隐式主张**（「ta 不喜欢我」「ta 只是感激我」…）
2. **为每个主张找证据和替代解释**（不让你只看自己想看的）
3. **检查信息对称性**（ta 知道你知道的那些事吗？）
4. **检测矛盾**（你的结论和你的行为是不是在打架？）
5. **基于心理学理论给出诊断**（不套模板，只挑机制匹配的理论）

### vs 通用 Chatbot

| | Lucid | ChatGPT / DeepSeek |
|---|---|---|
| **输出结构** | 9 节点管线确保每个 claim 都经过证据、替代解释、信息对称性三重检查 | 单次 LLM 调用，无结构保证 |
| **幻觉控制** | Prompt 约束 → 代码校验 → retry_hint 三层防御；checklist 模式让 LLM 只做选择题不做数字题 | 依赖 prompt 措辞，无代码层防御 |
| **多视角验证** | evidence + alternative + info_symmetry 三个独立节点交叉验证 | 单一回复视角，无内置对立面检查 |
| **理论引用** | RAG 检索 10 张心理学卡片，特异性优先匹配（不强行套模板） | 靠训练数据记忆，常把一切归因于「焦虑型依恋」 |
| **多轮记忆** | MemorySaver 自动续上下文 + SqliteStore 持久化画像与行为 fact | 无状态或仅上下文窗口内记忆 |
| **路由决策** | supervisor 纯代码路由（非 LLM），不受 LLM 随机性影响 | 依赖 prompt 或 LLM 自行判断，不可控 |

## Demo 演示

> 🚧 截图待补充

## 管线架构

```
                    ┌──────────────┐
                    │  preprocess  │  文本 → claims + identity + direction
                    └──┬────────┬──┘
                       │        │
              ┌────────┘        └────────┐
              ▼                          ▼
      ┌──────────────┐          ┌───────────────┐
      │  signals     │          │  supervisor   │  纯代码路由，不调 LLM
      │  (并行支线)  │          └───────┬───────┘
      └──────────────┘                  │
                                ┌───────┘
                                ▼
                         ┌──────────────┐
                         │ compression  │  behaviors 有界压缩（状态层）
                         └──────┬───────┘
                         ┌──────┼──────────────┐
                         ▼      ▼              ▼
                  ┌──────────┐  ┌──────────────┐  ┌─────────────┐
                  │ evidence │  │ alternative  │  │ info_       │
                  │ 证据收集 │  │ 替代解释     │  │ symmetry    │
                  └────┬─────┘  └──────┬───────┘  │ 信息对称性  │
                       │              │           └──────┬──────┘
                       └──────────────┼──────────────────┘
                                      ▼
                          ┌──────────────────────┐
                          │  contradictory       │  汇总 weak_evidence
                          │  registration        │  + alternative gaps
                          └──────────┬───────────┘
                                     ▼
                          ┌──────────────────────┐
                          │  summary            │  三段诊断 + 行动建议
                          │  (记忆写入 Store)    │  + 长期记忆持久化
                          └──────────────────────┘
```

### 9 个节点

| 节点 | 做什么 |
|------|--------|
| **preprocess** | 文本拆解为 claims，提取身份画像，判定 `direction`（single / both） |
| **signals** | 并行运行 — 提取用户和 ta 的行为信号，量化评分，行为出生即写 fact |
| **supervisor** | 纯代码路由 — 根据 pending claims 决定下一步调用哪些分析节点 |
| **compression** | 状态层有界压缩 — behaviors 总数超阈值时压缩最老，保留最近 N + 统计摘要 |
| **evidence** | 为每个 claim 搜索支持证据，计算 `credit_score` |
| **alternative** | 为每个 claim 生成替代解释，判定 claim 类型（事实 vs 解读） |
| **info_symmetry** | 只对 `direction="both"` 的 claims 执行 — 检查双方信息是否对等 |
| **contradictory** | 汇总弱证据和替代解释缺口，标记 `type` 区分来源 |
| **summary** | 三段结构诊断（偏差 → 机制 → 行动），写入画像 + timeline |

## 核心技术特点

### 1. 状态层有界压缩（不是 prompt 层补丁）

`all_signals.behaviors` 只在拼 prompt 时压副本的话，state 里的数据依然无限增长、撑爆 checkpoint 内存。Lucid 用独立的 **compression_node** 在状态层做有界化：behaviors 总数超过阈值（默认 200）时，把最老的行为压成「分组统计摘要」，保留最近 50 条原始记录。

**量化**：300 条行为 → 51 条（摘要 + 最近 50），上下文体积 **-83%**。

### 2. 画像 / 观察分离 + 记忆检索化

Lucid 的上下文分三类，用三种机制处理：

| 类型 | 例子 | 机制 |
|------|------|------|
| **动态画像** | 主动性/情感表达/信号清晰度三个分数 | 加权累积，跨会话恢复 |
| **静态画像** | 年龄、学校、关系阶段 | LLM 提取，每次注入 prompt |
| **观察** | 具体行为（邀约、拒绝、主动联系） | 出生即写 fact，语义检索注入 |

历史行为不再整块灌回 state，而是转成 fact 存入 Store，分析时用 BGE 语义检索 top-K 相关事实注入——被压缩掉的行为以语义可检索的形式保留在长期记忆里，而非永久丢失。

### 3. 身份画像

preprocess 用 LLM 从对话中提取用户和 ta 的静态身份信息（年龄/学校/关系阶段等），字段级累积、跨会话持久，每次分析都注入——让诊断「合乎身份」，而不是套模板。

### 4. RAG 知识增强（理论卡 + 记忆 fact 双检索）

- **理论卡**：10 张心理学卡片，BGE embedding → numpy 余弦检索。`common_misinterpretations`（否定语义）不嵌入，避免污染向量方向
- **记忆 fact**：行为事实同样走 BGE + numpy 检索，与理论卡共用同一个 embedding 模型

### 5. JSON 稳定性（response_format + checklist）

- **Checklist 模式**：LLM 回答选择题，代码查表算分，不直接让 LLM 拍数字
- **`response_format=json_object`**：API 层强制输出合法 JSON，显著减少长 JSON 输出的犹豫/自我修正，解析失败率下降
- **三层解析 fallback**：`json.loads` → 正则提取代码块 → 找首尾花括号

### 6. SSE 流式输出

分析过程通过 SSE 实时推送：**9 个节点的进度**（「正在分析行为信号」「正在收集证据」…）+ **summary 逐字输出**（打字机效果）。前端用 `fetch + ReadableStream` 接收，等待不再干瞪眼。

## 技术栈

- **编排**：LangGraph StateGraph + 条件路由
- **模型**：DeepSeek v4-flash (T=0)
- **后端**：FastAPI（SSE 流式端点 `/stream`）
- **前端**：React 19 + Vite（产品级界面，SSE 流式接收）
- **嵌入**：BGE-base-zh-v1.5（sentence_transformers，理论卡 + fact 共享单例）
- **检索**：numpy 余弦相似度（无需外部向量数据库）
- **记忆**：MemorySaver + SqliteStore（画像 + fact + timeline）
- **评估**：RAG recall + 结构完整性 + LLM 裁判覆盖度 + 正确性 MCQ

## 评估指标

基于 10 个 golden dataset 场景（easy/medium/hard）：

| 指标 | 分数 | 说明 |
|------|:---:|------|
| RAG Recall@1 | 0.60 | 首位检索命中率 |
| RAG Recall@3 | 0.93 | 前三位命中率 |
| 结构完整性 | 1.00 | graph 是否跑完、各节点是否产出 |
| 覆盖度（裁判） | 0.74 | LLM 裁判对分析覆盖度的评判 *（受限 RAG 卡片仅 10 张）* |
| 正确性（裁判） | 0.96 | 3 道 MCQ → 代码查表算分 |
| **加权总分** | **0.91** | — |

**v3 优化量化**：上下文压缩 **-83%**（300 条行为 → 51 条摘要 + 最近 50 条）。

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/mstiandi/lucid.git
cd lucid

# 2. 安装后端依赖
pip install langgraph langchain langchain-openai fastapi uvicorn httpx sentence_transformers

# 3. 设置 API Key
export DEEPSEEK_API_KEY=sk-your-key-here

# 4. 启动后端（SSE 流式）
python -m uvicorn achievement_graph.thought_v1.api:app --host 127.0.0.1 --port 8000

# 5. 启动前端（另开终端）
cd frontend
npm install
npm run dev
# → 打开 http://localhost:5173
```

## 项目结构

```
lucid/
├── achievement_graph/thought_v1/    # 主管线
│   ├── graph.py                     # 图结构 + 编译
│   ├── api.py                       # FastAPI + SSE 流式端点
│   ├── app.py                       # 旧 Gradio 入口（已弃用）
│   ├── nodes/                       # 9 个节点的实现
│   └── state/JokerState.py          # 状态定义 + TypedDict
├── tools/
│   ├── llm/                         # LLM 调用 + JSON 提取
│   ├── context/                     # Token 预算 + 压缩 + prompt 构建
│   ├── memory/sqlite_store.py       # 长期记忆
│   ├── rag/                         # BGE 嵌入 + 理论卡/fact 双检索
│   ├── eval/                        # 评估框架 (scorer + judge + runner)
│   └── reducer/                     # State 合并 helper
├── frontend/                        # React + Vite 前端
│   └── src/App.jsx                  # SSE 接收 + 进度条 + 打字机
└── prompts/                         # 各节点的 System Prompt (.md)
```

## 已知局限

- **同模型裁判偏袒**：用 DeepSeek 评判 DeepSeek 的输出存在隐性宽容，理想方案是用 GPT-4o-mini 做交叉裁判
- **RAG 卡片少**：10 张卡片够用但覆盖面有限，1000 张以上需考虑换 ChromaDB 等外部向量数据库
- **依赖 sentence_transformers**：首次运行需下载 BGE 模型（~400MB），轻量部署可注释 RAG 模块
- **alternative 节点 JSON 偶发失败**：长复杂 JSON 输出的固有 LLM 局限，`response_format` 已缓解但未根治

## License

MIT
