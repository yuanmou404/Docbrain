"""Cross-encoder 重排（BAAI/bge-reranker-v2-m3，sentence-transformers）。"""
import logging
import os
import threading

from app.config import Settings

logger = logging.getLogger("docbrain.reranker")
_lock = threading.Lock()


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.reranker_model
        self.enabled = settings.reranker_enabled
        self._ce = None  # 懒加载（首次调用时）

    def _ensure(self):
        if self._ce is not None or not self.enabled:
            return
        with _lock:
            if self._ce is not None:
                return
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415

                # 国内网络 / 代理环境下直连 huggingface.co 常失败，走可配置镜像
                endpoint = getattr(self.settings, "hf_endpoint", "") or ""
                if endpoint:
                    os.environ.setdefault("HF_ENDPOINT", endpoint.rstrip("/"))
                logger.info(
                    "加载重排模型 %s（首次运行需下载权重，endpoint=%s）...",
                    self.model_name,
                    os.environ.get("HF_ENDPOINT", "官方源"),
                )
                self._ce = CrossEncoder(self.model_name)
            except Exception as e:  # noqa: BLE001
                logger.warning("重排模型不可用，回退纯混合检索: %s", e)
                self.enabled = False

    def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        """返回与 passages 等长的相关性得分列表；不可用时返回 None。"""
        if not self.enabled or not passages:
            return None if not passages else []
        self._ensure()
        if self._ce is None:
            return None
        scores = self._ce.predict([(query, p) for p in passages])
        return [float(s) for s in scores]
