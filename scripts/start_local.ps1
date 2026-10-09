# DocBrain · Windows 本地一键启动（无需 Docker）
# 用法：在 DocBrain 目录下执行  powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1
# 前置：本机已运行 Ollama(11434, 含 bge-m3 与 qwen2.5) 与 Qdrant(6333)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== DocBrain 本地启动 ==" -ForegroundColor Cyan

# ── 1. 找一个真实可用的 Python（跳过 Microsoft Store 占位别名）──────────
function Test-RealPython($cmd) {
    try {
        $v = & $cmd --version 2>&1
        return ($LASTEXITCODE -eq 0) -and ($v -match "Python \d+\.\d+")
    } catch { return $false }
}

$py = $null
foreach ($cand in @("python", "py", "python3")) {
    $c = Get-Command $cand -ErrorAction SilentlyContinue
    if ($c -and (Test-RealPython $c.Source)) { $py = $c.Source; break }
}
if (-not $py) {
    # 兜底：本机托管 Python
    $managed = "$env:USERPROFILE\.workbuddy\binaries\python\versions\3.13.12\python.exe"
    if (Test-Path $managed) { $py = $managed }
}
if (-not $py) {
    Write-Host "未找到可用的 Python，请先安装 Python 3.11+ 并勾选 Add to PATH" -ForegroundColor Red
    Write-Host "下载地址：https://www.python.org/downloads/windows/" -ForegroundColor Yellow
    exit 1
}
Write-Host "Python: $py" -ForegroundColor Gray

# ── 2. 创建虚拟环境 ────────────────────────────────────────────────────
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "创建虚拟环境 .venv ..." -ForegroundColor Yellow
    & $py -m venv .venv
}
$venvPy = ".\.venv\Scripts\python.exe"

# ── 3. 安装依赖（已装则跳过）────────────────────────────────────────────
$needInstall = -not (& $venvPy -c "import fastapi, qdrant_client, fitz" 2>$null; $?)
if ($needInstall) {
    Write-Host "安装 torch(CPU) 与项目依赖 ..." -ForegroundColor Yellow
    & $venvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $venvPy -m pip install -r requirements.txt
} else {
    Write-Host "依赖已就绪，跳过安装" -ForegroundColor Gray
}

# ── 4. 检查外部服务 ────────────────────────────────────────────────────
foreach ($svc in @(@{n="Ollama"; u="http://localhost:11434/api/tags"}, @{n="Qdrant"; u="http://localhost:6333/collections"})) {
    try {
        Invoke-WebRequest -Uri $svc.u -UseBasicParsing -TimeoutSec 5 -Proxy "http://x" -ErrorAction SilentlyContinue | Out-Null
        Write-Host "$($svc.n) 已就绪" -ForegroundColor Green
    } catch {
        try {
            $r = Invoke-WebRequest -Uri $svc.u -UseBasicParsing -TimeoutSec 5
            Write-Host "$($svc.n) 已就绪" -ForegroundColor Green
        } catch {
            Write-Host "$($svc.n) 未运行（$($svc.u)）——未启动时问答会失败" -ForegroundColor Yellow
        }
    }
}

# ── 5. 启动 ────────────────────────────────────────────────────────────
Write-Host "`n启动服务：http://localhost:8000" -ForegroundColor Cyan
& $venvPy -m uvicorn app.main:app --host 0.0.0.0 --port 8000
