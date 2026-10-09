# DocBrain · 本地文档问答系统

全本地运行的文档问答系统，**数据不出机器**：解析、向量库（Qdrant）、推理（Ollama）与 Web UI 全部在本机。

---

## 🚀 启动方法（第一次用，只看这一节就够了）

支持 Windows / Linux / macOS，Python 3.11+（本项目实测 3.13）。Docker 是可选的，不装也能跑。

### 第 0 步：把两个外部服务跑起来（只需做一次）

DocBrain 依赖本机两个服务，必须先启动：

| 服务 | 端口 | 怎么装 | 怎么确认已就绪 |
|---|---|---|---|
| **Ollama**（嵌入 + 生成） | 11434 | 官网下载安装：<https://ollama.com/download/windows> | 终端执行 `ollama -v` 有版本输出 |
| **Qdrant**（向量库） | 6333 | GitHub Releases 下载 `x86_64-pc-windows-msvc.zip`：<https://github.com/qdrant/qdrant/releases>，解压后双击 `qdrant.exe` | 浏览器打开 <http://localhost:6333/collections> 返回 JSON |

Ollama 装好后拉两个模型（约 5GB，只做一次）：

```powershell
ollama pull bge-m3        # 嵌入模型
ollama pull qwen2.5:7b    # 生成模型（内存 ≤ 8GB 建议换 qwen2.5:3b）
ollama list               # 确认两个模型都在列表里
```

### 第 1 步：启动 DocBrain

**方式 A · 一条命令（推荐）**

打开 **PowerShell**，执行：

```powershell
cd <你的 DocBrain 目录>        # 例如 cd D:\Projects\DocBrain
powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1
```

脚本会自动完成：探测真实 Python（自动跳过 Microsoft Store 的假 `python`）→ 建 `.venv` → 装依赖 → 检查 Ollama / Qdrant → 启动服务。
首次执行要装 torch 等依赖，约 5~15 分钟；之后秒起。**窗口不要关，关了服务就停。**

**方式 B · 手动执行（想看清每一步时用）**

```powershell
cd <你的 DocBrain 目录>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> 若 `python -m venv .venv` 报「Python was not found」并弹出 Microsoft Store，说明系统只有占位别名，
> 改用真实解释器路径建环境，例如：
> `C:\Users\<你的用户名>\.workbuddy\binaries\python\versions\3.13.12\python.exe -m venv .venv`

**方式 C · Docker Compose（已装 Docker Desktop 才选）**

```bash
cp .env.example .env      # 可选：按需改模型 / 阈值
docker compose up -d      # 拉起 qdrant / ollama / ollama-init / api / web
```

首次会自动拉镜像并下载 bge-m3 + qwen2.5（较久），进度用 `docker compose logs -f ollama-init` 查看；就绪后开 <http://localhost:8080>。有 NVIDIA GPU 时取消 `docker-compose.yml` 中 ollama 服务的 GPU 注释可加速。

### 第 2 步：确认启动成功

```powershell
curl.exe --noproxy '*' http://127.0.0.1:8000/api/health
```

看到 `"ollama":true,"qdrant":true` 即正常。
（终端若配置了 HTTP 代理，访问 localhost **必须**加 `--noproxy '*'`，否则会报 502 / 连接被拒；浏览器访问不受影响。）

然后浏览器打开 **http://localhost:8000** —— Web UI 与 API 都由这一个端口提供（方式 C 走 nginx 则是 8080，API 文档在 `/docs`）。

### 第 3 步：开始使用

1. 左侧「文档库」上传 PDF / Word / Markdown，或粘贴网页 URL 导入；
   想先验证链路是否通畅，可直接上传仓库自带的 `examples/sample_doc.md`（里面附了几个测试问题）；
2. 等状态显示切分片段数（CPU 上每百片段约 30 秒，属正常）；
3. 右侧输入框提问，回答中的 `[n]` 可点击跳转原文片段与**页码**。

### 停止与重启

- **停止**：运行服务的终端按 `Ctrl + C`；或 `netstat -ano | findstr :8000` 查到 PID 后 `taskkill /F /PID <PID>`。
- **重启**：重新执行第 1 步的命令即可（依赖已装会直接启动）。

> ⚠️ 提问后提示「知识库为空 / 证据不足」时，**先看回答底部的统计**：
> `candidates=0` → 文档还没成功入库，回到第 3 步检查上传状态；
> `candidates>0` 但 `top_score` 很低 → 召回正常，是文档里确实没有支撑该问题的内容（换用文档中的说法提问，或调低 `DOCBRAIN_REFUSE_THRESHOLD`）。
> 详细说明见下方「`mode` 与拒答的三种情形」。

---

```
┌─────────────┐   ┌─────────────────────────── API (FastAPI) ───────────────────────────┐
│   Web UI    │──▶│  解析层            检索层                  生成层                    │
│ (nginx 静态) │   │  PDF(文本层/OCR)   BM25 (jieba) ─┐        RAG Prompt                │
└─────────────┘   │  Word / MD / 网页  Qdrant 向量 ──┼─RRF─▶  cross-encoder 重排 ─▶ Ollama LLM
                  │  语义切分(标题+句子)              ┘        引用溯源 [n] / 低置信拒答  │
                  └────────────────────────────────────────────────────────────────────┘
                          ▲                                   ▲
                   Qdrant 向量库                        Ollama (bge-m3 + qwen2.5)
```

## 核心能力与实现

| 简历条目 | 实现 |
|---|---|
| 多格式解析与语义切分 | `app/parsing/`：PDF（PyMuPDF，字号启发式标题识别）/ Word（python-docx，样式标题+表格）/ Markdown（标题层级+代码块保护）/ 网页（trafilatura 正文抽取）；`chunker.py` 按标题分节 + 句子边界滑动窗口切分，带重叠 |
| BM25 + 向量混合检索 | `bm25.py`（jieba 分词，rank-bm25）与 `vector.py`（Qdrant, bge-m3 1024 维）双路召回，`hybrid.py` 以 RRF（k=60）融合 |
| cross-encoder 重排 | `reranker.py` 加载 `BAAI/bge-reranker-v2-m3` 对 RRF 候选精排，取 top-N 进入生成 |
| 引用溯源 | 回答中每个结论带 `[n]` 编号，前端点击可查看原文片段、来源文档与**页码**（PDF 保留页级定位），并可下载原文档 |
| 低置信拒答 | 两级：① 重排最高分 < `REFUSE_THRESHOLD`（默认 0.08）直接拒答不调 LLM；② 提示词要求 `evidence_sufficient=false` 时明确说明无法确定。阈值按 bge-reranker-v2-m3 实测校准（无关问题 top-1 ≤ 0.015、可答问题 ≥ 0.13），换重排模型需重测 |
| 扫描件 OCR + LLM 纠错分支 | `ocr.py`：PDF 页文本层过少 → 判定扫描页 → rapidocr 识别 → LLM 纠错（形近字/0-O/断行等），`eval/ocr_field_eval.py` 做字段抽取准确率对比 |
| 100 条评测集 + Recall@5/MRR | `eval/build_dataset.py` LLM 生成问答集；`eval/run_eval.py` 切分粒度(256/512/1024) × 检索策略(bm25/vector/hybrid/hybrid+rerank) 网格评测，输出 Markdown 报告 |

## 评测流程

```bash
# 1. 上传若干文档（Web UI 或 curl，见下）
# 2. 生成 100 条问答评测集（LLM 生成问题，片段为 gold 答案）
python eval/build_dataset.py --target 100

# 3. 网格评测：切分 × 策略 → Recall@5 / MRR / 延迟，报告落盘 eval/reports/
python eval/run_eval.py                 # 全量网格
python eval/run_eval.py --no-rerank     # 无重排快速版

# 4.（可选）扫描件字段抽取对比：纯 OCR vs OCR + LLM 纠错
#    先把扫描件截图放入 eval/ocr_samples/ 并补全 ground_truth.csv
python eval/ocr_field_eval.py
```

> 跨切分粒度时 chunk_id 不一致，命中判定采用检索片段与 gold 片段的 **3-gram 字符重叠率 ≥ 0.5**。

## API 速览

```bash
# 上传文档
curl -F "file=@报告.pdf" http://localhost:8000/api/documents/upload

# 导入网页
curl -X POST http://localhost:8000/api/documents/web \
     -H 'Content-Type: application/json' -d '{"url":"https://example.com/article"}'

# 提问（限定某文档范围可选 doc_ids）
curl -X POST http://localhost:8000/api/query \
     -H 'Content-Type: application/json' \
     -d '{"question":"系统的三层架构是怎么设计的？"}'
```

问答响应结构：`answer`（含 `[n]` 引用标记）、`refused`（是否拒答）、`citations[]`（doc/页码/原文片段/得分）、
`candidates`（召回候选数）、`top_score`（最高重排分）、`refuse_threshold`（阈值）、`mode`、`timings`、`models`。

### `mode` 与拒答的三种情形

| mode | 含义 | 触发条件 |
|---|---|---|
| `rag` | 正常检索问答 | 召回片段最高分 ≥ 阈值，送 LLM 生成并标注引用 |
| `smalltalk` | 寒暄直答，不检索 | 问题匹配问候/自我介绍（你好、你是谁、谢谢…） |
| `refuse` | 拒答 | ① 知识库为空；② 双路召回 0 命中；③ 召回到片段但最高分 < 阈值 |

**排查"答不出来"时看 `candidates` 和 `top_score`**：

- `candidates=0` → 知识库里没有相关文档（最常见：文档还没上传，或上传失败）
- `candidates>0` 但 `top_score` 远低于阈值 → 召回正常，是文档中确实没有支撑该问题的内容；
  若确认文档里有，多为问法与原文用词差异过大，换用文档中的说法提问，或调低 `DOCBRAIN_REFUSE_THRESHOLD`
- 换重排模型必须重测阈值：取若干「文档内能答」与「完全无关」的问题各跑一遍 reranker，找两组 top-1 分数的分界（bge-reranker-v2-m3 实测：无关 ≤ 0.015、可答 ≥ 0.13，故取 0.08）

## 本地开发（不用 Docker）

启动步骤见文首「🚀 启动方法」的方式 A（一键脚本）/ 方式 B（手动）。默认已连本机 Ollama(11434) 与 Qdrant(6333)，
无需额外环境变量；需要改模型名、切分大小、拒答阈值等时，编辑 `.env`（也支持 `DOCBRAIN_` 前缀的环境变量覆盖）。

### 常见坑（Windows + Python 3.13 实测）

| 现象 | 原因 | 解决 |
|---|---|---|
| `Python was not found` / 弹出 Microsoft Store | 系统只有 Store 占位别名 | 用真实解释器路径建 venv：`C:\Users\<你>\.workbuddy\binaries\python\versions\3.13.12\python.exe -m venv .venv`；或到「设置→应用→应用执行别名」关掉 python.exe 的 Store 别名 |
| `pip install` 在 numpy 上报 `meson.build ERROR` | Python 3.13 无 numpy 2.0 预编译包，只能源码构建 | 已修正为 `numpy>=2.1,<3`（见 requirements.txt） |
| `No matching distribution found for rapidocr-onnxruntime` | 该包 1.3+ 不支持 Python 3.13 | 已换用官方后继包 `rapidocr>=3.0`（代码同时兼容旧包名） |
| 重排模型下载报 `CERTIFICATE_VERIFY_FAILED` | 代理 SSL 中间人 / 直连 huggingface.co 不通 | `.env` 中设 `DOCBRAIN_HF_ENDPOINT=https://hf-mirror.com`（已默认） |
| `curl http://localhost:8000` 返回 502/拒绝连接 | 终端走了 HTTP 代理 | 加 `--noproxy '*'`；浏览器访问不受影响 |
| 启动日志反复打印「等待 Qdrant 就绪（n/15）」 | Qdrant 没运行 | 双击 `qdrant.exe` 即可，服务会在 15 次重试内自动连上（约 90 秒窗口），无需重启 DocBrain |
| 日志出现「Ollama 未就绪」/ 回答报调用失败 | Ollama 没运行或模型没拉 | `ollama serve` 后 `ollama pull bge-m3` `ollama pull qwen2.5:7b` |

## 目录结构

```
DocBrain/
├── docker-compose.yml      # qdrant / ollama / ollama-init / api / web
├── Dockerfile              # API 镜像（CPU torch + 全依赖）
├── deploy/nginx.conf       # Web UI 静态 + /api 反代
├── app/
│   ├── main.py             # FastAPI 入口（生命周期、静态挂载）
│   ├── config.py           # 环境变量配置（DOCBRAIN_ 前缀）
│   ├── pipeline.py         # 入库管线 & 问答管线（引用/拒答）
│   ├── state.py            # 组件装配（重试、BM25 重建）
│   ├── api/                # documents / query 路由
│   ├── parsing/            # pdf / docx / md / web / ocr / chunker
│   ├── retrieval/          # bm25 / vector(qdrant) / hybrid(RRF) / reranker / embedder
│   └── llm/                # Ollama 客户端 & 提示词
├── web/index.html          # 单文件 Web UI
├── eval/                   # build_dataset / run_eval / ocr_field_eval
├── scripts/pull_models.py
└── data/                   # uploads/（原文） docs/（元数据+解析结果）
```

## 常见问题

- **首次问答很慢**：bge-reranker-v2-m3 权重（约 2GB）首次下载 + 加载；追求速度可 `DOCBRAIN_RERANKER_ENABLED=false`。
- **扫描件没走 OCR**：确认 `DOCBRAIN_OCR_ENABLED=true` 且 rapidocr 已安装（Docker 镜像内置）；日志中会打印「第 N 页判定为扫描件」。
- **修改了 chunk_size**：对已有文档不生效（历史索引按入库时参数切分），可删除文档重新上传。
- **内存/显存吃紧**：换 `qwen2.5:3b`（改 `.env` 的 `DOCBRAIN_LLM_MODEL` 后 `docker compose up -d ollama-init` 重拉）。

## 状态

- ✅ 多格式解析、混合检索、重排、引用溯源、拒答、Web UI、Compose 部署
- 🔬 进行中：OCR 字段抽取对比实验（`eval/ocr_field_eval.py`，待补真实扫描件样本）、检索网格评测报告（`eval/run_eval.py`，跑完后将 Recall@5 / MRR 数据回填简历）
