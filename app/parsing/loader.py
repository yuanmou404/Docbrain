"""多格式解析入口：按扩展名分发。"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.parsing.chunker import Block
from app.parsing.docx_parser import parse_docx
from app.parsing.md_parser import parse_markdown_file, parse_text_file
from app.parsing.ocr import OCREngine
from app.parsing.pdf_parser import parse_pdf
from app.parsing.web_parser import parse_url

logger = logging.getLogger("docbrain.loader")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


@dataclass
class ParsedDoc:
    name: str
    blocks: list[Block] = field(default_factory=list)
    pages: int | None = None
    ocr_pages: int = 0
    title: str | None = None


def parse_file(path: str, settings: Settings, ocr: OCREngine) -> ParsedDoc:
    ext = Path(path).suffix.lower()
    name = Path(path).name

    if ext == ".pdf":
        blocks, meta = parse_pdf(path, settings, ocr)
    elif ext == ".docx":
        blocks, meta = parse_docx(path)
    elif ext in (".md", ".markdown"):
        blocks, meta = parse_markdown_file(path)
    elif ext == ".txt":
        blocks, meta = parse_text_file(path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}（支持 {', '.join(sorted(SUPPORTED_EXTENSIONS))}）")

    return ParsedDoc(
        name=name,
        blocks=blocks,
        pages=meta.get("pages"),
        ocr_pages=meta.get("ocr_pages", 0),
        title=meta.get("name"),
    )


def parse_web(url: str) -> ParsedDoc:
    blocks, meta = parse_url(url)
    return ParsedDoc(
        name=meta.get("name") or url,
        blocks=blocks,
        pages=meta.get("pages"),
        ocr_pages=meta.get("ocr_pages", 0),
        title=meta.get("name"),
    )


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", name)
