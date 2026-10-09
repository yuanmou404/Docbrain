"""OCR 引擎封装：rapidocr（默认，纯 pip 依赖）→ pytesseract 兜底；
扫描件文本可选 LLM 纠错分支。"""
import io
import logging
from typing import Optional

import numpy as np

from app.llm.client import OllamaClient
from app.llm.prompts import OCR_CORRECT_SYSTEM
from app.config import Settings

logger = logging.getLogger("docbrain.ocr")


class OCREngine:
    def __init__(self, settings: Settings, ollama: Optional[OllamaClient] = None):
        self.settings = settings
        self.ollama = ollama
        self._kind: Optional[str] = None  # None=未探测, rapidocr/pytesseract/none
        self._rapid = None

    # ── 后端探测 ──────────────────────────────────────
    def _ensure(self) -> None:
        if self._kind is not None:
            return
        err: Optional[Exception] = None
        for module_path in ("rapidocr", "rapidocr_onnxruntime"):
            try:
                mod = __import__(module_path, fromlist=["RapidOCR"])
                self._rapid = mod.RapidOCR()
                self._kind = "rapidocr"
                logger.info("OCR 后端: %s", module_path)
                return
            except Exception as e:  # noqa: BLE001
                err = e
        logger.warning("rapidocr 不可用(%s)，尝试 pytesseract", err)
        try:
            import pytesseract  # noqa: F401,PLC0415

            self._kind = "pytesseract"
            logger.info("OCR 后端: pytesseract（需系统安装 tesseract 与 chi_sim 语言包）")
        except Exception as e2:  # noqa: BLE001
            logger.warning("pytesseract 也不可用(%s)，扫描件将跳过 OCR", e2)
            self._kind = "none"

    def available(self) -> bool:
        self._ensure()
        return self._kind in ("rapidocr", "pytesseract")

    # ── 图像 → 文本 ───────────────────────────────────
    def extract_png(self, png_bytes: bytes) -> str:
        self._ensure()
        if self._kind == "rapidocr":
            import cv2  # noqa: PLC0415

            arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
            result, _ = self._rapid(arr)
            if not result:
                return ""
            texts = []
            for item in result:
                if isinstance(item, dict):
                    texts.append(str(item.get("text", "")))
                else:  # [box, text, score]
                    texts.append(str(item[1]))
            return "\n".join(t for t in texts if t)
        if self._kind == "pytesseract":
            import pytesseract  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415

            img = Image.open(io.BytesIO(png_bytes))
            for lang in ("chi_sim+eng", "eng"):
                try:
                    return pytesseract.image_to_string(img, lang=lang)
                except Exception:  # noqa: BLE001
                    continue
            return ""
        return ""

    def extract_file(self, image_path) -> str:
        with open(image_path, "rb") as f:
            return self.extract_png(f.read())

    def extract_pdf_page(self, page) -> str:
        """page: fitz.Page。2 倍分辨率渲染后 OCR。"""
        import fitz  # noqa: PLC0415

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return self.extract_png(pix.tobytes("png"))

    # ── LLM 纠错分支 ──────────────────────────────────
    def correct(self, raw: str) -> str:
        if not self.settings.ocr_llm_correct or not raw.strip():
            return raw
        if not self.ollama or not self.ollama.is_up():
            return raw
        try:
            fixed = self.ollama.chat(
                messages=[
                    {"role": "system", "content": OCR_CORRECT_SYSTEM},
                    {"role": "user", "content": raw},
                ],
                temperature=0.0,
            ).strip()
            return fixed if fixed else raw
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM 纠错失败，保留 OCR 原文: %s", e)
            return raw
