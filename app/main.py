"""DocBrain FastAPI 入口：本地文档问答系统（数据不出机器）。"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import documents, query
from app.config import settings
from app.state import build_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("docbrain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 数据目录
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "uploads").mkdir(exist_ok=True)
    (settings.data_dir / "docs").mkdir(exist_ok=True)

    # 组件初始化（Qdrant 重试、BM25 重建等）
    app.state.ctx = build_context()
    logger.info("DocBrain 就绪: llm=%s embed=%s(%s) reranker=%s",
                settings.llm_model, settings.embed_model, settings.embed_provider,
                settings.reranker_enabled)
    yield


app = FastAPI(
    title="DocBrain",
    description="全本地运行的文档问答系统：多格式解析 / BM25+向量混合检索 / cross-encoder 重排 / 引用溯源 / 低置信拒答",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由（注册顺序优先于静态挂载）
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")

# Web UI（本地开发直连 8000 端口时也能用；Docker 下由 nginx 提供）
_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
