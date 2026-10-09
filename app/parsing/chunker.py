"""语义切分：按标题分节 + 句子边界滑动窗口，带重叠。"""
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Block:
    """解析器输出的最小结构单元（段落/标题/表格/OCR 页）。"""

    text: str
    page: Optional[int] = None
    heading: Optional[str] = None  # 非空表示该块本身是一个标题
    ocr: bool = False


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: Optional[int]
    heading: Optional[str]
    ocr: bool

    def to_payload(self, doc_id: str, doc_name: str) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page": self.page,
            "heading": self.heading,
            "text": self.text,
            "ocr": self.ocr,
        }


def split_sentences(text: str) -> list[str]:
    """中英文混合分句：换行 → 中英句末标点。"""
    out: list[str] = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        for seg in re.split(r"(?<=[。！？!?；;])", para):
            seg = seg.strip()
            if not seg:
                continue
            if len(seg) > 600:  # 超长英文段按句号再切
                for s2 in re.split(r"(?<=\.)\s+", seg):
                    s2 = s2.strip()
                    if s2:
                        out.append(s2)
            else:
                out.append(seg)
    return out


def _join(parts: list[str]) -> str:
    """中文段无缝拼接，英文段保留空格。"""
    if not parts:
        return ""
    buf = parts[0]
    for p in parts[1:]:
        sep = " " if (buf[-1].isascii() and p[0].isascii()) else ""
        buf += sep + p
    return buf


def chunk_blocks(
    blocks: list[Block],
    chunk_size: int = 512,
    overlap: int = 64,
    min_chars: int = 50,
) -> list[Chunk]:
    """把 Block 序列切成语义块。

    规则：
    1. 标题（heading 非空）开启新节，块不跨节；
    2. 节内按句子累积，超过 chunk_size 即成块；
    3. 相邻块之间保留尾部 overlap 字符的句子级重叠；
    4. 节末过短的块并入前一块（不丢内容）。
    """
    # 1) 展开为 (句子, 页码, ocr) 流，并按标题切段
    sections: list[tuple[Optional[str], list[tuple[str, Optional[int], bool]]]] = []
    cur_heading: Optional[str] = None
    cur: list[tuple[str, Optional[int], bool]] = []
    for b in blocks:
        if b.heading and b.heading != cur_heading and b.heading.strip():
            if cur:
                sections.append((cur_heading, cur))
            cur_heading, cur = b.heading.strip(), []
        for s in split_sentences(b.text):
            cur.append((s, b.page, b.ocr))
    if cur:
        sections.append((cur_heading, cur))

    # 2) 节内滑动窗口切分
    all_chunks: list[Chunk] = []
    for heading, items in sections:
        sec_chunks: list[list[tuple[str, Optional[int], bool]]] = []
        buf: list[tuple[str, Optional[int], bool]] = []
        for s, page, ocr in items:
            if not buf:
                buf = [(s, page, ocr)]
                continue
            if sum(len(x[0]) for x in buf) + len(s) > chunk_size:
                sec_chunks.append(buf)
                # 尾部重叠：从后往前取不超过 overlap 字符的完整句子
                tail: list[tuple[str, Optional[int], bool]] = []
                total = 0
                for x in reversed(buf):
                    if total + len(x[0]) > overlap:
                        break
                    tail.insert(0, x)
                    total += len(x[0])
                buf = tail + []  # 复制，避免共享引用
            buf.append((s, page, ocr))
        if buf:
            sec_chunks.append(buf)

        # 3) 节末短块合并
        merged: list[list[tuple[str, Optional[int], bool]]] = []
        for c in sec_chunks:
            text = _join([x[0] for x in c])
            if merged and len(text.strip()) < min_chars:
                merged[-1].extend(c)
            else:
                merged.append(c)

        for c in merged:
            text = _join([x[0] for x in c]).strip()
            if not text:
                continue
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text,
                    page=c[0][1],
                    heading=heading,
                    ocr=any(x[2] for x in c),
                )
            )
    return all_chunks


def blocks_asdict(blocks: list[Block]) -> list[dict]:
    return [asdict(b) for b in blocks]
