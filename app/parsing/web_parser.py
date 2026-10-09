"""网页解析：httpx 抓取 + trafilatura 正文抽取（bs4 兜底）。"""
import logging

import httpx

from app.parsing.chunker import Block

logger = logging.getLogger("docbrain.web")

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DocBrain/0.1"


def parse_url(url: str) -> tuple[list[Block], dict]:
    r = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": _UA})
    r.raise_for_status()
    html = r.text

    title = None
    extracted = None
    try:
        import trafilatura  # noqa: PLC0415

        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
        meta = trafilatura.extract_metadata(html)
        title = getattr(meta, "title", None) if meta else None
    except Exception as e:  # noqa: BLE001
        logger.warning("trafilatura 抽取失败(%s)，使用 bs4 兜底", e)

    if not extracted:
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415

            extracted = BeautifulSoup(html, "html.parser").get_text("\n")
        except Exception:  # noqa: BLE001
            extracted = html

    # 按行组织为段落块（过滤导航等短行）
    blocks = [
        Block(text=para.strip())
        for para in extracted.split("\n")
        if len(para.strip()) >= 20
    ]
    name = title or url
    return blocks, {"pages": None, "ocr_pages": 0, "name": name}
