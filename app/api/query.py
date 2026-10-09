"""问答路由：混合检索 + 重排 + 引用溯源 + 低置信拒答。"""
import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas import QueryRequest

logger = logging.getLogger("docbrain.api.query")

router = APIRouter(tags=["query"])


@router.post("/query")
def ask(body: QueryRequest, request: Request):
    ctx = request.app.state.ctx
    try:
        return ctx.query.answer(body.question, body.doc_ids, body.top_k)
    except Exception as e:  # noqa: BLE001
        logger.exception("问答失败: %s", body.question)
        raise HTTPException(500, f"问答失败: {e}") from e
