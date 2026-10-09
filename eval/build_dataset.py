#!/usr/bin/env python
"""自建问答评测集：从已入库文档中采样片段，用本地 LLM 生成问题（可 --no-llm 退化为启发式）。

用法（先启动服务并上传文档）：
    python eval/build_dataset.py --target 100 --out eval/dataset.jsonl

输出 jsonl 每行：
    {"id": 0, "question": "...", "doc_id": "...", "gold_chunk_ids": [...], "gold_text": "..."}
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.llm.client import OllamaClient  # noqa: E402
from app.llm.prompts import QA_GEN_PROMPT  # noqa: E402
from app.retrieval.vector import QdrantStore  # noqa: E402


def heuristic_question(p: dict) -> str:
    topic = p.get("heading") or p["text"][:24]
    return f"{p['doc_name']}中，关于「{topic.strip()}」的内容是什么？"


def main() -> None:
    ap = argparse.ArgumentParser(description="构建问答评测集")
    ap.add_argument("--target", type=int, default=100, help="目标条数（默认 100）")
    ap.add_argument("--out", default=str(ROOT / "eval" / "dataset.jsonl"))
    ap.add_argument("--no-llm", action="store_true", help="不调用 LLM，用启发式生成问题")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qdrant = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    payloads = qdrant.scroll_all_payloads()
    if not payloads:
        print("向量库为空：请先通过 Web UI 或 API 上传文档。")
        sys.exit(1)

    # 过滤过短片段，按文档分组轮转均匀采样
    rng = random.Random(args.seed)
    by_doc: dict[str, list[dict]] = {}
    for p in payloads:
        if len(p.get("text", "")) >= 120:
            by_doc.setdefault(p["doc_id"], []).append(p)
    for lst in by_doc.values():
        rng.shuffle(lst)

    sampled: list[dict] = []
    doc_ids = list(by_doc.keys())
    while len(sampled) < args.target and any(by_doc[d] for d in doc_ids):
        for d in doc_ids:
            if by_doc[d] and len(sampled) < args.target:
                sampled.append(by_doc[d].pop())
    print(f"可用片段 {len(payloads)}，采样 {len(sampled)} 条（覆盖 {len({p['doc_id'] for p in sampled})} 个文档）")

    ollama = None
    use_llm = not args.no_llm
    if use_llm:
        ollama = OllamaClient(settings.ollama_base_url)
        if not ollama.is_up() or not ollama.has_model(settings.llm_model):
            print("警告：Ollama/LLM 不可用，回退启发式生成")
            use_llm = False

    items = []
    for i, p in enumerate(sampled):
        if use_llm:
            try:
                q = ollama.chat(
                    [{"role": "user", "content": QA_GEN_PROMPT.format(text=p["text"][:1200])}],
                    model=settings.llm_model,
                    temperature=0.4,
                ).strip().splitlines()[0].strip()
                q = q.strip('"“”')
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}] LLM 生成失败({e})，退化为启发式")
                q = heuristic_question(p)
        else:
            q = heuristic_question(p)
        if len(q) < 6:
            q = heuristic_question(p)
        items.append(
            {
                "id": i,
                "question": q,
                "doc_id": p["doc_id"],
                "doc_name": p.get("doc_name", ""),
                "gold_chunk_ids": [p["chunk_id"]],
                "gold_text": p["text"],
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  已生成 {i + 1}/{len(sampled)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"评测集已写入 {out}（{len(items)} 条）")


if __name__ == "__main__":
    main()
