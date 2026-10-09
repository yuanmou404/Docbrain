"""混合检索编排：BM25 + 向量 → RRF 融合 → cross-encoder 重排。"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings
from app.retrieval.bm25 import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import Reranker
from app.retrieval.vector import QdrantStore

logger = logging.getLogger("docbrain.retrieval")


@dataclass
class ScoredChunk:
    chunk_id: str
    payload: dict
    score: float


@dataclass
class RetrieveResult:
    chunks: list[ScoredChunk] = field(default_factory=list)
    reranked: bool = False


def rrf_fuse(rank_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank)。"""
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, (cid, _score) in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        qdrant: QdrantStore,
        bm25: BM25Store,
        reranker: Reranker,
    ):
        self.settings = settings
        self.embedder = embedder
        self.qdrant = qdrant
        self.bm25 = bm25
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        doc_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        use_rerank: Optional[bool] = None,
    ) -> RetrieveResult:
        s = self.settings
        k = top_k or s.top_k
        cand = max(k * s.candidate_multiplier, k)

        # 1) 双路召回
        qv = self.embedder.encode_one(query)
        bm_hits = self.bm25.search(query, cand, doc_ids)  # [(id, score)]
        vec_hits = self.qdrant.search(qv, cand, doc_ids)  # [(id, score, payload)]

        # 2) RRF 融合
        fused = rrf_fuse(
            [bm_hits, [(cid, sc) for cid, sc, _ in vec_hits]], k=s.rrf_k
        )

        payload_map = {cid: pl for cid, _sc, pl in vec_hits}

        def payload_of(cid: str) -> dict:
            if cid in payload_map:
                return payload_map[cid]
            p = self.bm25.get(cid)
            return p or {"chunk_id": cid, "text": ""}

        shortlist = fused[: max(k * 2, k)]
        if not shortlist:
            return RetrieveResult([], False)

        # 3) cross-encoder 重排
        should_rerank = self.reranker.enabled if use_rerank is None else use_rerank
        if should_rerank:
            texts = [payload_of(cid)["text"] for cid, _ in shortlist]
            scores = self.reranker.rerank(query, texts)
            if scores:
                ranked = sorted(zip(shortlist, scores), key=lambda x: -x[1])
                chunks = [
                    ScoredChunk(chunk_id=cid, payload=payload_of(cid), score=score)
                    for (cid, _), score in ranked[:k]
                ]
                return RetrieveResult(chunks, reranked=True)
            logger.info("重排不可用，回退 RRF 排序")

        chunks = [ScoredChunk(cid, payload_of(cid), sc) for cid, sc in shortlist[:k]]
        return RetrieveResult(chunks, reranked=False)
