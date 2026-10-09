"""DocBrain 全局配置（环境变量前缀 DOCBRAIN_，支持 .env 文件）。"""
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "DOCBRAIN_", "env_file": ".env", "extra": "ignore"}

    # ── 存储 ──────────────────────────────────────────
    data_dir: Path = Path("data")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docbrain"

    # ── 推理服务 ──────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.2
    llm_num_ctx: int = 8192

    # ── 嵌入 ──────────────────────────────────────────
    embed_provider: str = "ollama"  # ollama | local
    embed_model: str = "bge-m3"
    local_embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # ── 重排 ──────────────────────────────────────────
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # HuggingFace 镜像（国内网络 / 代理证书拦截时改用 hf-mirror）
    hf_endpoint: str = "https://hf-mirror.com"

    # ── 切分 ──────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_chars: int = 50

    # ── 检索 ──────────────────────────────────────────
    top_k: int = 8
    rrf_k: int = 60
    rerank_top_n: int = 4
    candidate_multiplier: int = 4

    # ── 拒答 ──────────────────────────────────────────
    # 阈值需按所用 cross-encoder 的打分区间校准：bge-reranker-v2-m3 实测
    # 无关问题 top-1 ≤ 0.015、可答问题 ≥ 0.13，故取 0.08；换模型需重测
    refuse_threshold: float = 0.08

    # ── OCR ───────────────────────────────────────────
    ocr_enabled: bool = True
    ocr_min_text_chars: int = 24
    ocr_llm_correct: bool = True


settings = Settings()
