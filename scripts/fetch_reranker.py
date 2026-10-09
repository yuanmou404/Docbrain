"""拉取 cross-encoder 重排模型权重到本地目录。

用途：当 huggingface.co 直连不通（国内网络 / 代理 SSL 拦截）时，
用镜像站把权重下载到 data/models/ 下，再由 DocBrain 以本地路径加载，
之后完全离线可用。

用法：
    python scripts/fetch_reranker.py                       # 默认 BAAI/bge-reranker-v2-m3
    python scripts/fetch_reranker.py BAAI/bge-reranker-base
    python scripts/fetch_reranker.py --endpoint https://hf-mirror.com

下载完成后把 .env 里的模型改成本地路径：
    DOCBRAIN_RERANKER_MODEL=./data/models/bge-reranker-v2-m3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "BAAI/bge-reranker-v2-m3"
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# 只取推理必需文件，跳过 README / 图片 / git 元数据
NEEDED_SUFFIX = (
    ".json",
    ".txt",
    ".model",
    ".safetensors",
    ".bin",
)
SKIP_PREFIX = ("assets/", ".git")


def list_files(endpoint: str, repo: str) -> list[str]:
    url = f"{endpoint}/api/models/{repo}"
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    siblings = r.json().get("siblings", [])
    out = []
    for item in siblings:
        name = item.get("rfilename", "")
        if not name or name.startswith(SKIP_PREFIX):
            continue
        if not name.endswith(NEEDED_SUFFIX):
            continue
        out.append(name)
    return out


def download(endpoint: str, repo: str, filename: str, dest: Path) -> bool:
    """返回 True 表示文件已就绪（新下载或已存在且非空）。"""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ 已存在，跳过: {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    url = f"{endpoint}/{repo}/resolve/main/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, 4):
        try:
            with httpx.stream("GET", url, timeout=60, follow_redirects=True) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                got = 0
                t0 = time.time()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            pct = got / total * 100
                            mb = got / 1e6
                            print(f"\r  ↓ {filename}: {pct:5.1f}% ({mb:.0f}/{total / 1e6:.0f} MB)", end="", flush=True)
            print()
            if got == 0:
                raise RuntimeError("下载内容为空（镜像返回 0 字节）")
            tmp.replace(dest)
            print(f"  ✓ 完成: {filename} ({got / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"\n  ✗ 第 {attempt} 次失败: {type(e).__name__}: {str(e)[:120]}")
            tmp.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 cross-encoder 重排模型权重到本地")
    ap.add_argument("repo", nargs="?", default=DEFAULT_REPO, help=f"模型仓库，默认 {DEFAULT_REPO}")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"镜像源，默认 {DEFAULT_ENDPOINT}")
    ap.add_argument("--out", default=None, help="输出目录，默认 data/models/<模型名>")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "data" / "models" / args.repo.split("/")[-1]
    print(f"仓库: {args.repo}\n镜像: {args.endpoint}\n目标: {out_dir}\n")

    try:
        files = list_files(args.endpoint, args.repo)
    except Exception as e:  # noqa: BLE001
        print(f"获取文件清单失败: {e}", file=sys.stderr)
        return 1

    if not files:
        print("未找到任何模型文件，请检查仓库名或镜像可用性", file=sys.stderr)
        return 1

    print(f"待下载 {len(files)} 个文件:")
    ok = True
    for name in files:
        ok = download(args.endpoint, args.repo, name, out_dir / name) and ok

    if not ok:
        print("\n部分文件下载失败，可重跑本脚本（已下载的文件会自动跳过）")
        return 1

    print(f"\n权重已就绪: {out_dir}")
    print(f"请在 .env 中设置: DOCBRAIN_RERANKER_MODEL={out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
