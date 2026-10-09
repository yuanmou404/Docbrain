#!/usr/bin/env python
"""检索横向评测：切分粒度 × 检索策略 网格，输出 Recall@5 / MRR / 延迟报告。

用法（需先 build_dataset.py 生成评测集，服务保持运行以提供嵌入/重排模型）：
    python eval/run_eval.py                       # 默认 256/512/1024 × bm25/vector/hybrid/hybrid+rerank
    python eval/run_eval.py --sizes 512 --strategies hybrid hybrid+rerank
    python eval/run_eval.py --no-rerank           # 跳过重排（无 GPU/提速）

命中判定：检索片段与 gold 片段的 3-gram 字符重叠率 ≥ 0.5 视为命中
（跨切分粒度时 chunk_id 不一致，用文本重叠代替 ID 精确匹配）。
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.llm.client import OllamaClient  # noqa: E402
from app.parsing.chunker import Block, chunk_blocks  # noqa: E402
from app.retrieval.bm25 import BM25Store  # noqa: E402
from app.retrieval.embedder import Embedder  # noqa: E402
from app.retrieval.hybrid import rrf_fuse  # noqa: E402
from app.retrieval.reranker import Reranker  # noqa: E402
from app.retrieval.vector import QdrantStore  # noqa: E402


def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> set:
        s = s.strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not gb:
        return 0.0
    return len(ga & gb) / len(gb)


def load_dataset(path: str) -> list[dict]:
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        print("评测集为空：请先运行 eval/build_dataset.py")
        sys.exit(1)
    return items


def load_doc_blocks() -> list[dict]:
    """读取入库时保存的解析结果（data/docs/*.json 中的 blocks）。"""
    docs_dir = settings.data_dir / "docs"
    out = []
    for f in sorted(docs_dir.glob("*.json")):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            if meta.get("blocks"):
                out.append(meta)
        except Exception as e:  # noqa: BLE001
            print(f"警告：跳过 {f.name}: {e}")
    return out


def evaluate(strategy_name: str, retrieve_fn, dataset: list[dict], k: int = 5) -> dict:
    recalls, mrrs, lats = [], [], []
    for item in dataset:
        t0 = time.perf_counter()
        top_texts = retrieve_fn(item["question"], k)
        lats.append((time.perf_counter() - t0) * 1000)
        hit_rank = None
        for rank, t in enumerate(top_texts, 1):
            if ngram_overlap(t, item["gold_text"]) >= 0.5:
                hit_rank = rank
                break
        recalls.append(1.0 if hit_rank else 0.0)
        mrrs.append(1.0 / hit_rank if hit_rank else 0.0)
    return {
        "strategy": strategy_name,
        "recall@5": round(statistics.mean(recalls), 4),
        "mrr": round(statistics.mean(mrrs), 4),
        "latency_ms": round(statistics.mean(lats), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="切分 × 检索策略 网格评测")
    ap.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.jsonl"))
    ap.add_argument("--sizes", type=int, nargs="*", default=[256, 512, 1024], help="切分粒度（字符数）")
    ap.add_argument("--strategies", nargs="*", default=["bm25", "vector", "hybrid", "hybrid+rerank"])
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    docs = load_doc_blocks()
    if not docs:
        print("data/docs 下没有已入库文档的解析结果，请先上传文档。")
        sys.exit(1)
    print(f"评测集 {len(dataset)} 条；文档 {len(docs)} 篇")

    ollama = OllamaClient(settings.ollama_base_url)
    if not ollama.is_up():
        print("Ollama 不可用（向量检索需要嵌入模型）")
        sys.exit(1)
    embedder = Embedder(settings, ollama)
    reranker = Reranker(settings)
    if args.no_rerank:
        reranker.enabled = False

    strategies = [s for s in args.strategies if s != "hybrid+rerank"] if args.no_rerank else args.strategies
    results: list[dict] = []

    for size in args.sizes:
        print(f"\n===== 切分粒度 {size} =====")
        # 1) 重切分
        payloads = []
        for meta in docs:
            blocks = [Block(**b) for b in meta["blocks"]]
            for c in chunk_blocks(blocks, size, settings.chunk_overlap, settings.min_chunk_chars):
                payloads.append(c.to_payload(meta["doc_id"], meta["name"]))
        print(f"  重切分完成：{len(payloads)} 个片段")

        # 2) 建临时索引
        t0 = time.perf_counter()
        vectors = embedder.encode_batch([p["text"] for p in payloads])
        col = f"{settings.qdrant_collection}_eval_{size}"
        store = QdrantStore(settings.qdrant_url, col, settings.embed_dim)
        store.drop_collection()
        store.ensure_collection()
        store.upsert_chunks(payloads, vectors)
        bm25 = BM25Store()
        bm25.build(payloads)
        payload_by_id = {p["chunk_id"]: p for p in payloads}
        print(f"  索引就绪（嵌入+入库 {time.perf_counter() - t0:.1f}s）")

        # 3) 预计算问题向量（跨策略复用）
        qvecs = embedder.encode_batch([it["question"] for it in dataset])

        def bm25_retrieve(q: str, k: int, _bm25=bm25):
            return [payload_by_id[cid]["text"] for cid, _ in _bm25.search(q, k)]

        _qmap = {it["question"]: i for i, it in enumerate(dataset)}

        def vector_retrieve(q: str, k: int, _store=store, _qvecs=qvecs, _qmap=_qmap):  # noqa: F811
            vec = _qvecs[_qmap[q]]
            return [pl["text"] for _cid, _sc, pl in _store.search(vec, k)]

        def hybrid_retrieve(q: str, k: int, _bm25=bm25, _store=store, _qvecs=qvecs, _qmap=_qmap):
            vec = _qvecs[_qmap[q]]
            bm = _bm25.search(q, k * 4)
            vc = [(cid, sc) for cid, sc, _ in _store.search(vec, k * 4)]
            fused = rrf_fuse([bm, vc], k=settings.rrf_k)[:k]
            return [payload_by_id[cid]["text"] for cid, _ in fused]

        def hybrid_rerank_retrieve(q: str, k: int, _bm25=bm25, _store=store, _qvecs=qvecs, _qmap=_qmap, _rer=reranker):
            vec = _qvecs[_qmap[q]]
            bm = _bm25.search(q, k * 4)
            vc = [(cid, sc) for cid, sc, _ in _store.search(vec, k * 4)]
            fused = rrf_fuse([bm, vc], k=settings.rrf_k)[: k * 3]
            texts = [payload_by_id[cid]["text"] for cid, _ in fused]
            scores = _rer.rerank(q, texts)
            if scores:
                ranked = sorted(zip(texts, scores), key=lambda x: -x[1])
                return [t for t, _ in ranked[:k]]
            return texts[:k]

        fns = {
            "bm25": bm25_retrieve,
            "vector": vector_retrieve,
            "hybrid": hybrid_retrieve,
            "hybrid+rerank": hybrid_rerank_retrieve,
        }

        # 4) 跑各策略
        for name in strategies:
            if name not in fns:
                print(f"  跳过未知策略 {name}")
                continue
            t0 = time.perf_counter()
            row = evaluate(f"chunk={size}/{name}", fns[name], dataset)
            row["chunk_size"] = size
            results.append(row)
            print(f"  {name:<14} Recall@5={row['recall@5']:.3f}  MRR={row['mrr']:.3f}  延迟={row['latency_ms']:.0f}ms  (总 {time.perf_counter() - t0:.1f}s)")

        store.drop_collection()

    # 5) 输出报告
    if not results:
        print("没有可用的评测结果")
        return
    best = max(results, key=lambda r: r["mrr"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = ROOT / "eval" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# DocBrain 检索评测报告",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 评测集：{len(dataset)} 条问答，{len(docs)} 篇文档",
        f"- 参数：overlap={settings.chunk_overlap}, top-k 候选=5, 命中判定=3-gram 重叠率≥0.5",
        f"- 重排器：{'启用(' + settings.reranker_model + ')' if reranker.enabled else '关闭'}",
        "",
        "| 切分粒度 | 检索策略 | Recall@5 | MRR | 平均延迟(ms) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['chunk_size']} | {r['strategy'].split('/', 1)[1]} | {r['recall@5']:.3f} | {r['mrr']:.3f} | {r['latency_ms']:.0f} |")
    lines += [
        "",
        f"**最优组合**：chunk={best['chunk_size']} / {best['strategy'].split('/', 1)[1]}（Recall@5={best['recall@5']:.3f}, MRR={best['mrr']:.3f}）",
        "",
    ]
    md_path = report_dir / f"eval_report_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = report_dir / f"eval_report_{ts}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存：{md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
