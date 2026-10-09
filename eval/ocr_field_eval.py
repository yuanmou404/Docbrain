#!/usr/bin/env python
"""OCR 字段抽取准确率对比：纯 OCR vs OCR + LLM 纠错【进行中——简历条目对应实验】

实验设计：
    对同一批扫描件截图，分别用「纯 OCR」与「OCR + LLM 纠错」两条分支提取关键字段，
    与人工标注的 ground truth 对比，输出各字段准确率对比报告。

准备材料：
    eval/ocr_samples/*.png|jpg     扫描件截图
    eval/ocr_samples/ground_truth.csv   标注文件，列：file,code,name,price
                                       （首次运行会自动生成模板与示例）

用法（需服务运行以调用 Ollama 纠错；--no-llm 可只测纯 OCR 基线）：
    python eval/ocr_field_eval.py
    python eval/ocr_field_eval.py --samples eval/ocr_samples --out eval/reports
"""
import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.llm.client import OllamaClient  # noqa: E402
from app.parsing.ocr import OCREngine  # noqa: E402

FIELDS = ["code", "name", "price"]

EXTRACT_PROMPT = """从下面的 OCR 文本中提取字段，输出一个 JSON 对象（缺失字段填 null，不要猜测）：
{{"code": "6位股票代码", "name": "证券名称", "price": "价格数字"}}

OCR 文本：
{text}"""

CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*元?")


def extract_fields_regex(text: str) -> dict:
    code = CODE_RE.search(text)
    price = None
    for m in PRICE_RE.finditer(text):
        price = m.group(1)
        break
    return {"code": code.group(1) if code else None, "name": None, "price": price}


def extract_fields_llm(text: str, ollama: OllamaClient) -> dict | None:
    try:
        content = ollama.chat(
            [{"role": "user", "content": EXTRACT_PROMPT.format(text=text[:1500])}],
            model=settings.llm_model,
            temperature=0.0,
        )
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        return {f: str(obj.get(f)) if obj.get(f) is not None else None for f in FIELDS}
    except Exception:  # noqa: BLE001
        return None


def normalize(field: str, v: str | None) -> str | None:
    if v is None:
        return None
    v = str(v).strip()
    if field == "code":
        v = re.sub(r"\D", "", v)[:6]
    elif field == "price":
        v = v.replace(",", "").replace("元", "")
    return v or None


def make_template(samples_dir: Path) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    gt = samples_dir / "ground_truth.csv"
    if not gt.exists():
        with gt.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "code", "name", "price"])
            w.writerow(["# 示例：把扫描件截图放进本目录，并按此格式补充标注", "", "", ""])
            w.writerow(["sample_01.png", "600000", "浦发银行", "8.52"])
        print(f"已生成标注模板：{gt}\n请放入扫描件截图并补全标注后重新运行。")
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR 字段抽取准确率对比")
    ap.add_argument("--samples", default=str(ROOT / "eval" / "ocr_samples"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    ap.add_argument("--no-llm", action="store_true", help="只测纯 OCR 基线（不做纠错对比）")
    args = ap.parse_args()

    samples_dir = Path(args.samples)
    gt_path = make_template(samples_dir)
    rows = []
    with gt_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("file") and not row["file"].startswith("#") and row.get("code"):
                rows.append(row)
    if not rows:
        print("标注文件为空，请先补全 ground_truth.csv")
        sys.exit(1)

    ollama = OllamaClient(settings.ollama_base_url)
    ocr = OCREngine(settings, ollama)
    if not ocr.available():
        print("OCR 引擎不可用：请安装 rapidocr-onnxruntime（或系统 tesseract）")
        sys.exit(1)

    use_llm = not args.no_llm and ollama.is_up() and ollama.has_model(settings.llm_model)
    print(f"样本 {len(rows)} 条；LLM 纠错分支：{'启用' if use_llm else '关闭'}")

    per_field_hit = {"raw": {f: [] for f in FIELDS}, "corrected": {f: [] for f in FIELDS}}
    details = []

    for i, row in enumerate(rows, 1):
        img = samples_dir / row["file"]
        if not img.exists():
            print(f"  [{i}] 缺图跳过: {row['file']}")
            continue
        raw = ocr.extract_file(img)
        corrected = ocr.correct(raw) if use_llm else raw

        def extract(text: str) -> dict:
            if use_llm:
                got = extract_fields_llm(text, ollama)
                if got:
                    return got
            got = extract_fields_regex(text)
            return got

        raw_fields = extract(raw)
        corr_fields = extract(corrected) if use_llm else raw_fields

        for f in FIELDS:
            gold = normalize(f, row.get(f))
            per_field_hit["raw"][f].append(1 if normalize(f, raw_fields.get(f)) == gold and gold else 0)
            per_field_hit["corrected"][f].append(
                1 if normalize(f, corr_fields.get(f)) == gold and gold else 0
            )
        details.append(
            {
                "file": row["file"],
                "raw_text_head": raw[:120],
                "corrected_text_head": corrected[:120],
                "raw_fields": raw_fields,
                "corrected_fields": corr_fields,
            }
        )
        print(f"  [{i}/{len(rows)}] {row['file']} 完成")

    # ── 报告 ──
    n = len(details)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = [
        "# OCR 字段抽取准确率对比报告",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}；样本 {n} 条",
        f"- 分支：A=纯 OCR；B=OCR + LLM 纠错（{settings.llm_model}）",
        "",
        "| 字段 | A 纯OCR 准确率 | B OCR+LLM纠错 准确率 | 提升 |",
        "|---|---|---|---|",
    ]
    for f in FIELDS:
        a = statistics.mean(per_field_hit["raw"][f]) if per_field_hit["raw"][f] else 0.0
        b = statistics.mean(per_field_hit["corrected"][f]) if per_field_hit["corrected"][f] else 0.0
        lines.append(f"| {f} | {a:.2%} | {b:.2%} | {(b - a):+.2%} |")
    overall_a = statistics.mean([x for f in FIELDS for x in per_field_hit["raw"][f]])
    overall_b = statistics.mean([x for f in FIELDS for x in per_field_hit["corrected"][f]])
    lines += [
        "",
        f"**总体**：A {overall_a:.2%} → B {overall_b:.2%}（{(overall_b - overall_a):+.2%}）",
        "",
        "> 说明：样本量较小时结论仅供参考；建议 ≥ 100 条样本后再填入简历数据。",
    ]

    md_path = out_dir / f"ocr_field_eval_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"ocr_field_eval_{ts}.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已保存：{md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
