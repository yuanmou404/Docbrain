"""Markdown / 纯文本解析：标题层级识别 + 代码块保护。"""
import re
from pathlib import Path

from app.parsing.chunker import Block

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_markdown_text(text: str) -> tuple[list[Block], dict]:
    blocks: list[Block] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if body:
            blocks.append(Block(text=body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            in_fence = not in_fence
            buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue
        m = _HEADING.match(stripped)
        if m:
            flush()
            title = m.group(2).strip()
            # 标题自身作为一个块（同时作为切分锚点）
            blocks.append(Block(text=title, heading=title))
            continue
        buf.append(line)
    flush()
    return blocks, {"pages": None, "ocr_pages": 0}


def parse_markdown_file(path: str) -> tuple[list[Block], dict]:
    return parse_markdown_text(Path(path).read_text(encoding="utf-8", errors="ignore"))


def parse_text_file(path: str) -> tuple[list[Block], dict]:
    """纯文本：按空行分段。"""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = [Block(text=para.strip()) for para in re.split(r"\n\s*\n", text) if para.strip()]
    return blocks, {"pages": None, "ocr_pages": 0}
