"""嵌入提供者：Ollama bge-m3（默认）或容器内 sentence-transformers。"""
import logging
from typing import Optional

from app.config import Settings
from app.llm.client import OllamaClient

logger = logging.getLogger("docbrain.embedder")


class Embedder:
    def __init__(self, settings: Settings, ollama: Optional[OllamaClient]):
        self.settings = settings
        self.ollama = ollama
        self._st = None
        if settings.embed_provider == "local":
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("加载本地嵌入模型 %s ...", settings.local_embed_model)
            self._st = SentenceTransformer(settings.local_embed_model)

    def encode_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []
        if self._st is not None:
            vecs = self._st.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
            return [v.tolist() for v in vecs]
        # Ollama 路径：分批避免请求体过大
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self.ollama.embed(texts[i : i + batch_size], self.settings.embed_model))
        return out

    def encode_one(self, text: str) -> list[float]:
        return self.encode_batch([text])[0]
