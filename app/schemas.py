"""API 请求/响应数据模型。"""
from typing import Optional

from pydantic import BaseModel, Field


class WebIngestRequest(BaseModel):
    url: str = Field(..., description="要导入的网页 URL")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    doc_ids: Optional[list[str]] = Field(None, description="限定检索的文档 ID 列表；None 表示全部文档")
    top_k: Optional[int] = Field(None, ge=1, le=32, description="候选片段数")


class Citation(BaseModel):
    label: int = Field(..., description="回答中的引用编号 [n]")
    doc_id: str
    doc_name: str
    page: Optional[int] = Field(None, description="原文页码（无页码的格式为 null）")
    chunk_id: str
    text: str = Field(..., description="被引用的原文片段")
    score: float = Field(..., description="相关性得分（重排后）")


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool = Field(..., description="是否拒答（证据不足）")
    citations: list[Citation] = []
    retrieved: int = Field(0, description="送入 LLM 的片段数（拒答时为 0）")
    candidates: int = Field(0, description="检索到的候选片段数（拒答时也能看到真实召回量）")
    top_score: Optional[float] = Field(None, description="候选最高重排得分（拒答时用于展示差距）")
    refuse_threshold: Optional[float] = Field(None, description="当前拒答阈值")
    mode: str = Field("rag", description="rag=正常问答 / refuse=拒答 / smalltalk=寒暄直答")
    reranked: bool = False
    timings: dict = Field(default_factory=dict)
    models: dict = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    chunks: int = 0
    pages: Optional[int] = None
    ocr_pages: int = 0
    created_at: str = ""
    upload_path: Optional[str] = None
