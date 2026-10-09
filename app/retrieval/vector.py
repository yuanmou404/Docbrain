"""Qdrant 向量库封装。"""
import logging

from qdrant_client import QdrantClient, models

logger = logging.getLogger("docbrain.qdrant")


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int):
        self.collection = collection
        self.dim = dim
        self.client = QdrantClient(url=url, timeout=120)

    def ensure_collection(self) -> None:
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection not in names:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE
                ),
            )
            logger.info("已创建向量集合 %s (dim=%d)", self.collection, self.dim)
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001
            pass  # 已存在时忽略

    def upsert_chunks(self, payloads: list[dict], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(id=p["chunk_id"], vector=v, payload=p)
            for p, v in zip(payloads, vectors)
        ]
        for i in range(0, len(points), 64):
            self.client.upsert(collection_name=self.collection, points=points[i : i + 64])

    def search(
        self,
        query_vector: list[float],
        k: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        """返回 [(chunk_id, score, payload)]。"""
        qf = None
        if doc_ids:
            qf = models.Filter(
                should=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=d))
                    for d in doc_ids
                ]
            )
        res = self.client.query_points(
            self.collection,
            query=query_vector,
            limit=k,
            query_filter=qf,
            with_payload=True,
        )
        return [(str(p.id), p.score, dict(p.payload or {})) for p in res.points]

    def scroll_all_payloads(self) -> list[dict]:
        out: list[dict] = []
        offset = None
        while True:
            pts, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True
            )
            out.extend(dict(p.payload or {}) for p in pts)
            if offset is None:
                break
        return out

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete_by_filter(
            self.collection,
            models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                ]
            ),
        )

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def drop_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection)
        except Exception:  # noqa: BLE001
            pass
