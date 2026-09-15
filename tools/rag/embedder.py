"""
共享 BGE embedding 工具：theory_store（理论卡检索）和 fact_retriever（记忆 fact 检索）
共用同一个 SentenceTransformer 实例，避免重复加载 ~400MB 模型。

模块级单例。import 时主线程加载（避免在 LangGraph 线程池中首次加载模型）。
"""
import os
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    SentenceTransformer = None

_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
_model = None

# BGE v1.5 非对称检索：query 侧加 instruction 前缀，passage 侧不加
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _resolve_model_path() -> str:
    """定位模型路径：本地缓存 → 自动下载 → 返回绝对路径。"""
    hub_cache = Path(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    )
    model_dir = hub_cache / "models--BAAI--bge-base-zh-v1.5"
    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snapshots:
            if (snap / "config.json").exists():
                return str(snap)
    from huggingface_hub import snapshot_download
    return snapshot_download(_MODEL_NAME)


def get_model():
    """懒加载 BGE model（模块级单例）。sentence_transformers 不可用时抛异常。"""
    global _model
    if not _AVAILABLE:
        raise RuntimeError("sentence_transformers 未安装，embedding 不可用")
    if _model is None:
        _model = SentenceTransformer(_resolve_model_path())
    return _model


def embed(texts) -> "object":
    """对文本列表做 L2 归一化 embedding，返回 numpy 数组（passage 侧，不加 instruction）。"""
    return get_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


def embed_query(query: str) -> "object":
    """query 侧 embedding：加 instruction 前缀（BGE v1.5 非对称检索），返回 (1, dim)。"""
    return get_model().encode(
        [_QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
