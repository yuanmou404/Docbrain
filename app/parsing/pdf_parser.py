"""PDF 解析：PyMuPDF 文本层 + 字号启发式标题识别 + 扫描页 OCR 分支。"""
import logging
import statistics
from typing import Optional

import fitz

from app.config import Settings
from app.parsing.chunker import Block
from app.parsing.ocr import OCREngine

logger = logging.getLogger("docbrain.pdf")


def _iter_line_sizes(page_dict: dict) -> list[float]:
    sizes = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if spans:
                sizes.append(max(sp["size"] for sp in spans))
    return sizes


def parse_pdf(path: str, settings: Settings, ocr: OCREngine) -> tuple[list[Block], dict]:
    doc = fitz.open(path)
    page_dicts = [p.get_text("dict") for p in doc]

    # 第一遍：全文字号中位数（标题判定的基准）
    all_sizes: list[float] = []
    for d in page_dicts:
        all_sizes.extend(_iter_line_sizes(d))
    median_size = statistics.median(all_sizes) if all_sizes else 10.0

    blocks: list[Block] = []
    ocr_pages = 0

    for pno, page in enumerate(doc):
        page_no = pno + 1
        raw_len = len(page.get_text("text").strip())

        # 扫描页：文本层几乎无字 → OCR + LLM 纠错分支
        if settings.ocr_enabled and raw_len < settings.ocr_min_text_chars:
            text = ""
            if ocr.available():
                text = ocr.extract_pdf_page(page).strip()
                if text:
                    text = ocr.correct(text)
            if text:
                blocks.append(Block(text=text, page=page_no, ocr=True))
                ocr_pages += 1
                logger.info("第 %d 页判定为扫描件，走 OCR 分支", page_no)
            elif raw_len == 0:
                logger.info("第 %d 页无可提取文本", page_no)
            continue

        # 文本层页：按 PDF block（段落）组织，字号启发式识别标题
        d = page_dicts[pno]
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            line_texts: list[str] = []
            max_size = 0.0
            for line in b.get("lines", []):
                t = "".join(sp["text"] for sp in line.get("spans", [])).strip()
                if not t:
                    continue
                line_texts.append(t)
                spans = line.get("spans", [])
                if spans:
                    max_size = max(max_size, max(sp["size"] for sp in spans))
            text = "\n".join(line_texts).strip()
            if not text:
                continue
            is_heading = (
                max_size >= median_size * 1.25
                and len(text) <= 50
                and not text.endswith(("。", "，", "；", ".", ",", ";"))
            )
            blocks.append(Block(text=text, page=page_no, heading=text if is_heading else None))

    doc.close()
    return blocks, {"pages": len(page_dicts), "ocr_pages": ocr_pages}
