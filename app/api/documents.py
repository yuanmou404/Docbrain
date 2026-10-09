"""文档管理路由：上传 / URL 导入 / 列表 / 删除 / 原文下载 / 健康检查。"""
import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.schemas import WebIngestRequest

logger = logging.getLogger("docbrain.api.documents")

router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def _ctx(request: Request):
    return request.app.state.ctx


@router.get("/health")
def health(request: Request):
    ctx = _ctx(request)
    return {
        "status": "ok",
        "ollama": ctx.ollama.is_up(),
        "qdrant": True,
        "reranker_enabled": ctx.reranker.enabled,
        "documents": len(ctx.ingest.list_docs()),
        "chunks": ctx.bm25.__len__(),
    }


@router.post("/documents/upload")
async def upload_document(file: UploadFile, request: Request):
    ctx = _ctx(request)
    filename = file.filename or "untitled"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型 {ext}，支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    data = await file.read()
    try:
        return await run_in_threadpool(ctx.ingest.ingest_file, data, filename)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("入库失败: %s", filename)
        raise HTTPException(500, f"入库失败: {e}") from e


@router.post("/documents/web")
def ingest_web(body: WebIngestRequest, request: Request):
    ctx = _ctx(request)
    try:
        return ctx.ingest.ingest_url(body.url)
    except Exception as e:  # noqa: BLE001
        logger.exception("网页导入失败: %s", body.url)
        raise HTTPException(400, f"网页导入失败: {e}") from e


@router.get("/documents")
def list_documents(request: Request):
    return _ctx(request).ingest.list_docs()


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, request: Request):
    ctx = _ctx(request)
    ctx.ingest.delete_doc(doc_id)
    return {"deleted": doc_id}


@router.get("/documents/{doc_id}/file")
def download_original(doc_id: str, request: Request):
    path = _ctx(request).ingest.get_upload_file(doc_id)
    if not path or not path.exists():
        raise HTTPException(404, "原文文件不存在（网页导入的文档无本地文件）")
    return FileResponse(path, filename=path.name.split("_", 1)[-1])
