"""应用上下文：生命周期内构建并持有各组件单例。"""
import logging
import time
from dataclasses import dataclass

from app.config import Settings, settings
from app.llm.client import OllamaClient
from app.parsing.ocr import OCREngine
from app.pipeline import IngestService, QueryService
from app.retrieval.bm25 import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector import QdrantStore

logger = logging.getLogger("docbrain.state")


@dataclass
class AppContext:
    settings: Settings
    ollama: OllamaClient
    embedder: Embedder
    qdrant: QdrantStore
    bm25: BM25Store
    reranker: Reranker
    ocr: OCREngine
    retriever: HybridRetriever
    ingest: IngestService
    query: QueryService


def build_context() -> AppContext:
    # 1) Ollama（本地推理服务）
    ollama = OllamaClient(settings.ollama_base_url)
    if not ollama.is_up():
        logger.error(
            "Ollama 未就绪（%s）——问答/嵌入将不可用。"
            "请确认推理服务已启动，或运行 docker compose up -d ollama ollama-init",
            settings.ollama_base_url,
        )
    else:
        for role, model in (("嵌入", settings.embed_model), ("生成", settings.llm_model)):
            if not ollama.has_model(model):
                logger.warning("Ollama 中未找到%s模型 %s，请执行: python scripts/pull_models.py", role, model)

    # 2) Qdrant（向量库，容器启动可能稍慢 → 重试）
    qdrant = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    last_err = None
    for i in range(15):
        try:
            qdrant.ensure_collection()
            last_err = None
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("等待 Qdrant 就绪（%d/15）: %s", i + 1, e)
            time.sleep(2)
    if last_err is not None:
        raise RuntimeError(f"Qdrant 无法连接: {last_err}")

    # 3) 嵌入 / 重排 / OCR
    embedder = Embedder(settings, ollama)
    reranker = Reranker(settings)
    ocr = OCREngine(settings, ollama)

    # 4) BM25 索引：启动时从向量库回放全量 payload 重建
    bm25 = BM25Store()
    payloads = qdrant.scroll_all_payloads()
    if payloads:
        bm25.build(payloads)
    logger.info("BM25 索引就绪：%d 个片段（%d 个文档）", len(bm25), len({p["doc_id"] for p in payloads}))

    retriever = HybridRetriever(settings, embedder, qdrant, bm25, reranker)
    ingest = IngestService(settings, ollama, embedder, qdrant, bm25, ocr)
    query = QueryService(settings, ollama, retriever)

    return AppContext(
        settings=settings,
        ollama=ollama,
        embedder=embedder,
        qdrant=qdrant,
        bm25=bm25,
        reranker=reranker,
        ocr=ocr,
        retriever=retriever,
        ingest=ingest,
        query=query,
    )
