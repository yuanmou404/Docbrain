#!/usr/bin/env python
"""拉取 DocBrain 所需的 Ollama 模型（嵌入 + 生成）。

用法：
    python scripts/pull_models.py                        # 默认本机 11434
    python scripts/pull_models.py --models bge-m3 qwen2.5:7b-instruct
"""
import argparse

import httpx


def main() -> None:
    ap = argparse.ArgumentParser(description="拉取 Ollama 模型")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--models", nargs="*", default=["bge-m3", "qwen2.5:7b"])
    args = ap.parse_args()

    for model in args.models:
        print(f"==> 拉取 {model}（大模型可能需要较长时间）...")
        try:
            r = httpx.post(
                f"{args.base_url}/api/pull",
                json={"model": model, "stream": False},
                timeout=httpx.Timeout(7200.0, connect=10.0),
            )
            if r.status_code == 200:
                print(f"    完成：{model}")
            else:
                print(f"    失败({r.status_code})：{r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"    失败：{e}")
    print("==> 全部完成")


if __name__ == "__main__":
    main()
