"""Ollama 本地推理服务客户端（对话 / 嵌入 / 模型拉取）。"""
import logging
from typing import Optional

import httpx

logger = logging.getLogger("docbrain.ollama")


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))

    # ── 基础状态 ──────────────────────────────────────
    def is_up(self) -> bool:
        try:
            return self._http.get(f"{self.base_url}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False

    def installed_models(self) -> list[str]:
        try:
            r = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            return [m.get("model", "") for m in r.json().get("models", [])]
        except Exception:
            return []

    def has_model(self, model: str) -> bool:
        models = self.installed_models()
        # Ollama 返回的名称可能带 ":latest" 后缀
        return any(m == model or m.split(":")[0] == model.split(":")[0] for m in models)

    # ── 对话 ──────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if options:
            payload["options"] = options
        r = self._http.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    # ── 嵌入 ──────────────────────────────────────────
    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        """批量文本嵌入。优先 /api/embed（Ollama >= 0.3），失败则回退逐条 /api/embeddings。"""
        try:
            r = self._http.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
            if r.status_code == 200:
                embs = r.json().get("embeddings")
                if embs and len(embs) == len(texts):
                    return embs
        except Exception as e:  # noqa: BLE001
            logger.warning("/api/embed 批量嵌入失败，回退单条接口: %s", e)

        out = []
        for t in texts:
            r = self._http.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": t},
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out

    # ── 模型拉取 ──────────────────────────────────────
    def pull(self, model: str) -> bool:
        try:
            r = self._http.post(
                f"{self.base_url}/api/pull",
                json={"model": model, "stream": False},
                timeout=httpx.Timeout(7200.0, connect=10.0),
            )
            return r.status_code == 200
        except Exception as e:  # noqa: BLE001
            logger.error("拉取模型 %s 失败: %s", model, e)
            return False
