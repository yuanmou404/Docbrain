"""核心业务管线：文档入库（解析→切分→嵌入→索引）与问答（检索→重排→生成→引用溯源）。"""
import json
import logging
import re
import time
import uuid as uuidlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.llm.client import OllamaClient
from app.llm.prompts import build_rag_messages
from app.parsing import loader
from app.parsing.chunker import blocks_asdict, chunk_blocks
from app.parsing.ocr import OCREngine
from app.retrieval.bm25 import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import QdrantStore

logger = logging.getLogger("docbrain.pipeline")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class IngestService:
    def __init__(
        self,
        settings: Settings,
        ollama: OllamaClient,
        embedder: Embedder,
        qdrant: QdrantStore,
        bm25: BM25Store,
        ocr: OCREngine,
    ):
        self.settings = settings
        self.ollama = ollama
        self.embedder = embedder
        self.qdrant = qdrant
        self.bm25 = bm25
        self.ocr = ocr

    # ── 入口 ──────────────────────────────────────────
    def ingest_file(self, data: bytes, filename: str) -> dict:
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"文件过大（>{MAX_UPLOAD_BYTES // 1024 // 1024}MB）")
        doc_id = uuidlib.uuid4().hex
        safe = loader.safe_filename(filename)
        upload_path = self.settings.data_dir / "uploads" / f"{doc_id}_{safe}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(data)

        parsed = loader.parse_file(str(upload_path), self.settings, self.ocr)
        return self._ingest(doc_id, parsed.name, parsed.blocks, parsed.pages, parsed.ocr_pages, str(upload_path))

    def ingest_url(self, url: str) -> dict:
        parsed = loader.parse_web(url)
        doc_id = uuidlib.uuid4().hex
        name = parsed.title or url
        return self._ingest(doc_id, name, parsed.blocks, parsed.pages, parsed.ocr_pages, None, source_url=url)

    # ── 入库主流程 ────────────────────────────────────
    def _ingest(
        self,
        doc_id: str,
        name: str,
        blocks: list,
        pages: Optional[int],
        ocr_pages: int,
        upload_path: Optional[str],
        source_url: Optional[str] = None,
    ) -> dict:
        t0 = time.perf_counter()
        s = self.settings

        chunks = chunk_blocks(blocks, s.chunk_size, s.chunk_overlap, s.min_chunk_chars)
        if not chunks:
            raise ValueError("解析后未得到有效文本内容")

        payloads = [c.to_payload(doc_id, name) for c in chunks]
        vectors = self.embedder.encode_batch([p["text"] for p in payloads])

        self.qdrant.upsert_chunks(payloads, vectors)
        self.bm25.add(payloads)

        meta = {
            "doc_id": doc_id,
            "name": name,
            "source_url": source_url,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "chunks": len(chunks),
            "pages": pages,
            "ocr_pages": ocr_pages,
            "blocks": blocks_asdict(blocks),  # 保留解析结果，供评测脚本重切分
        }
        if upload_path:
            meta["upload_path"] = upload_path
        meta_file = self.settings.data_dir / "docs" / f"{doc_id}.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        elapsed = round(time.perf_counter() - t0, 2)
        logger.info("入库完成 %s: %d 块 / %d 页(OCR %d 页) / %.1fs", name, len(chunks), pages or 0, ocr_pages, elapsed)
        return {k: v for k, v in meta.items() if k != "blocks"} | {"elapsed_s": elapsed}

    # ── 管理 ──────────────────────────────────────────
    def list_docs(self) -> list[dict]:
        out = []
        docs_dir = self.settings.data_dir / "docs"
        if not docs_dir.exists():
            return out
        for f in sorted(docs_dir.glob("*.json")):
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                out.append(
                    {k: v for k, v in meta.items() if k != "blocks"}
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("读取文档元数据失败 %s: %s", f, e)
        out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return out

    def delete_doc(self, doc_id: str) -> bool:
        meta_file = self.settings.data_dir / "docs" / f"{doc_id}.json"
        upload_files = list((self.settings.data_dir / "uploads").glob(f"{doc_id}_*"))
        self.qdrant.delete_doc(doc_id)
        self.bm25.delete_doc(doc_id)
        if meta_file.exists():
            meta_file.unlink()
        for f in upload_files:
            f.unlink(missing_ok=True)
        return True

    def get_upload_file(self, doc_id: str) -> Optional[Path]:
        files = list((self.settings.data_dir / "uploads").glob(f"{doc_id}_*"))
        return files[0] if files else None


class QueryService:
    def __init__(self, settings: Settings, ollama: OllamaClient, retriever: HybridRetriever):
        self.settings = settings
        self.ollama = ollama
        self.retriever = retriever

    # ── LLM 输出解析 ──────────────────────────────────
    @staticmethod
    def _parse_llm_json(content: str) -> dict:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
        return {"answer": text.strip(), "evidence_sufficient": True}

    def _refuse(
        self,
        question: str,
        reason: str,
        timings: dict,
        candidates: int = 0,
        top_score: Optional[float] = None,
        reranked: bool = False,
    ) -> dict:
        """拒答响应。candidates/top_score 保留真实召回信息，便于用户判断
        是「知识库没这篇文档」还是「阈值定得偏高」。"""
        return {
            "question": question,
            "answer": reason,
            "refused": True,
            "citations": [],
            "retrieved": 0,
            "candidates": candidates,
            "top_score": top_score,
            "refuse_threshold": self.settings.refuse_threshold if reranked else None,
            "mode": "refuse",
            "reranked": reranked,
            "timings": timings,
            "models": self._models_info(),
        }

    # ── 寒暄 / 无关指令的直答（不走检索，避免"你好"也被判证据不足）────
    _SMALLTALK = re.compile(
        r"^(你好|您好|hi|hello|hey|在吗|在么|哈喽|嗨|早上好|中午好|下午好|晚上好|"
        r"你是谁|你叫什么|介绍一下你自己|你能做什么|你会什么|谢谢|感谢|thanks|thank you|"
        r"再见|拜拜|bye)[!！。,.，~～?？\s]*$",
        re.I,
    )

    def _smalltalk_reply(self, question: str) -> Optional[str]:
        q = question.strip()
        if len(q) > 24 or not self._SMALLTALK.match(q):
            return None
        if re.match(r"^(你是谁|你叫什么|介绍一下你自己|你能做什么|你会什么)", q):
            return (
                "我是 DocBrain，一个完全本地运行的文档问答助手："
                "文档解析、向量检索与回答生成都在本机完成，数据不出机器。"
                "请把 PDF / Word / Markdown 上传到左侧文档库，或导入网页，"
                "然后就可以针对文档内容提问了（回答会标注引用片段与页码）。"
            )
        return "你好，我是 DocBrain 本地文档问答助手。请上传文档后向我提问，我会基于文档内容回答并标注引用来源。"

    def _models_info(self) -> dict:
        return {
            "llm": self.settings.llm_model,
            "embed": self.settings.embed_model,
            "reranker": self.settings.reranker_model if self.retriever.reranker.enabled else None,
        }

    # ── 问答主流程 ────────────────────────────────────
    def answer(self, question: str, doc_ids: Optional[list[str]] = None, top_k: Optional[int] = None) -> dict:
        timings: dict = {}
        t0 = time.perf_counter()

        # 0) 寒暄/自我介绍：不走检索，直接回复
        small = self._smalltalk_reply(question)
        if small:
            timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
            return {
                "question": question,
                "answer": small,
                "refused": False,
                "citations": [],
                "retrieved": 0,
                "candidates": 0,
                "top_score": None,
                "refuse_threshold": None,
                "mode": "smalltalk",
                "reranked": False,
                "timings": timings,
                "models": self._models_info(),
            }

        # 1) 混合检索 + 重排
        result = self.retriever.retrieve(question, doc_ids, top_k)
        timings["retrieve_ms"] = round((time.perf_counter() - t0) * 1000)
        candidates = len(result.chunks)
        top_score = round(result.chunks[0].score, 4) if result.chunks else None

        # 2) 低置信度 → 显式拒答（不调用 LLM，避免编造）
        # 2a) 知识库为空：先提示上传文档，而不是"换个问法"
        if doc_ids is None and len(self.retriever.bm25) == 0:
            return self._refuse(
                question,
                "知识库为空：当前没有任何已入库的文档片段。请先在左侧上传 PDF / Word / Markdown，"
                "或粘贴网页链接导入，然后再提问。",
                timings,
            )
        # 2b) 双路召回都没命中（问题与语料完全无交集）
        if not result.chunks:
            return self._refuse(
                question,
                "本次没有召回到任何候选片段，通常说明知识库里缺少与该问题相关的内容。"
                "请确认目标文档已上传，或换用文档里出现过的说法再问一次。",
                timings,
            )
        # 2c) 召回到了片段，但最高分低于阈值 → 说明文档中无支撑该问题的证据
        if result.reranked and result.chunks[0].score < self.settings.refuse_threshold:
            return self._refuse(
                question,
                f"检索到 {candidates} 个候选片段，但最高相关度仅 {top_score:.3f}，"
                f"低于拒答阈值 {self.settings.refuse_threshold}，说明当前文档中没有能支撑该问题的内容，"
                "因此拒答以避免编造。若该问题确实应能从文档中回答，请检查文档是否已完整入库，"
                "或换用文档中的原文表述提问。",
                timings,
                candidates=candidates,
                top_score=top_score,
                reranked=True,
            )

        # 3) 构建 RAG 上下文，调用本地 LLM
        context_chunks = [c.payload for c in result.chunks]
        messages = build_rag_messages(question, context_chunks)
        t_llm = time.perf_counter()
        try:
            content = self.ollama.chat(
                messages,
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                num_ctx=self.settings.llm_num_ctx,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("LLM 调用失败: %s", e)
            return self._refuse(
                question,
                f"本地推理服务调用失败：{e}。请确认 Ollama 正在运行且已拉取模型。",
                timings,
                candidates=candidates,
                top_score=top_score,
                reranked=result.reranked,
            )
        timings["llm_ms"] = round((time.perf_counter() - t_llm) * 1000)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000)

        # 4) 解析输出与引用溯源
        obj = self._parse_llm_json(content)
        answer_text = str(obj.get("answer", "")).strip() or content.strip()
        sufficient = bool(obj.get("evidence_sufficient", True))

        markers = []
        for m in re.findall(r"\[(\d+)\]", answer_text):
            n = int(m)
            if 1 <= n <= len(result.chunks) and n not in markers:
                markers.append(n)

        citations = []
        for n in markers:
            c = result.chunks[n - 1]
            citations.append(
                {
                    "label": n,
                    "doc_id": c.payload.get("doc_id", ""),
                    "doc_name": c.payload.get("doc_name", ""),
                    "page": c.payload.get("page"),
                    "chunk_id": c.chunk_id,
                    "text": c.payload.get("text", ""),
                    "score": round(c.score, 4),
                }
            )

        return {
            "question": question,
            "answer": answer_text,
            "refused": not sufficient,
            "citations": citations,
            "retrieved": len(result.chunks),
            "candidates": candidates,
            "top_score": top_score,
            "refuse_threshold": self.settings.refuse_threshold if result.reranked else None,
            "mode": "rag",
            "reranked": result.reranked,
            "timings": timings,
            "models": self._models_info(),
        }
