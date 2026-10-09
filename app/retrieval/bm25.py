"""BM25 关键词检索（jieba 分词），内存索引，增删后懒重建。"""
import logging
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

jieba.setLogLevel(logging.WARNING)
logger = logging.getLogger("docbrain.bm25")


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip() and t != " "]


class BM25Store:
    def __init__(self):
        self._payloads: dict[str, dict] = {}  # chunk_id -> payload
        self._order: list[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = False

    # ── 索引维护 ──────────────────────────────────────
    def build(self, payloads: list[dict]) -> None:
        self._payloads = {p["chunk_id"]: p for p in payloads}
        self._order = list(self._payloads.keys())
        self._dirty = True

    def add(self, payloads: list[dict]) -> None:
        for p in payloads:
            cid = p["chunk_id"]
            if cid not in self._payloads:
                self._order.append(cid)
            self._payloads[cid] = p
        self._dirty = True

    def delete_doc(self, doc_id: str) -> None:
        removed = {cid for cid, p in self._payloads.items() if p.get("doc_id") == doc_id}
        if removed:
            self._order = [cid for cid in self._order if cid not in removed]
            for cid in removed:
                self._payloads.pop(cid, None)
            self._dirty = True

    def _ensure(self) -> None:
        if self._dirty or self._bm25 is None:
            corpus = [tokenize(self._payloads[cid]["text"]) for cid in self._order]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._dirty = False

    # ── 检索 ──────────────────────────────────────────
    def search(
        self, query: str, k: int = 10, doc_ids: Optional[list[str]] = None
    ) -> list[tuple[str, float]]:
        """返回 [(chunk_id, bm25_score)]，按分值降序。"""
        self._ensure()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        pairs: list[tuple[str, float]] = []
        for idx, s in enumerate(scores):
            if s <= 0:
                continue
            cid = self._order[idx]
            p = self._payloads[cid]
            if doc_ids and p.get("doc_id") not in doc_ids:
                continue
            pairs.append((cid, float(s)))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def get(self, chunk_id: str) -> Optional[dict]:
        return self._payloads.get(chunk_id)

    def all_payloads(self) -> list[dict]:
        return [self._payloads[cid] for cid in self._order]

    def __len__(self) -> int:
        return len(self._payloads)
