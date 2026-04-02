
<#
.SYNOPSIS
    训练产物清理脚本（本地 Windows 机器运行）

.DESCRIPTION
    清理训练过程中产生的日志、模型缓存、回放文件、指标数据等。
    支持按类别选择性清理，也支持一键全部清理。

.PARAMETER All
    清理所有类别（日志 + 模型 + 回放 + 指标）

.PARAMETER Logs
    清理所有组件的运行时日志

.PARAMETER Models
    清理模型缓存（P2P ONNX + Checkpoint）

.PARAMETER Replay
    清理 TrainClient 的回放记录文件（.jsonl）

.PARAMETER Metrics
    清理 Learner 的 Dashboard 指标数据

.PARAMETER DryRun
    预览模式，只显示将要删除的文件，不实际删除

.EXAMPLE
    .\clean.ps1 -All                    # 清理全部
    .\clean.ps1 -Logs                   # 只清理日志
    .\clean.ps1 -Models -Replay         # 清理模型和回放
    .\clean.ps1 -All -DryRun            # 预览全部清理目标
#>

param(
    [switch]$All,
    [switch]$Logs,
    [switch]$Models,
    [switch]$Replay,
    [switch]$Metrics,
    [switch]$DryRun
)

# ---- 项目根目录（脚本所在目录）----
$ProjectRoot = $PSScriptRoot

# ---- 颜色输出辅助函数 ----
function Write-Header($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Write-Item($text)   { Write-Host "  $text" -ForegroundColor Gray }
function Write-Done($text)   { Write-Host "  [OK] $text" -ForegroundColor Green }
function Write-Skip($text)   { Write-Host "  [跳过] $text" -ForegroundColor Yellow }

# ---- 清理函数：删除指定目录下匹配模式的文件 ----
function Remove-MatchedFiles {
    param(
        [string]$Directory,       # 目标目录（绝对路径）
        [string]$Pattern,         # 文件匹配模式（如 *.log）
        [string]$Label,           # 显示标签
        [bool]$Recurse = $false   # 是否递归子目录
    )

    $fullPath = Join-Path $ProjectRoot $Directory

    if (-not (Test-Path $fullPath)) {
        Write-Skip "$Label - 目录不存在: $Directory"
        return
    }

    if ($Recurse) {
        $files = Get-ChildItem -Path $fullPath -Filter $Pattern -Recurse -File -ErrorAction SilentlyContinue
    } else {
        $files = Get-ChildItem -Path $fullPath -Filter $Pattern -File -ErrorAction SilentlyContinue
    }

    if ($files.Count -eq 0) {
        Write-Skip "$Label - 无匹配文件"
        return
    }

    $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
    $sizeStr = if ($totalSize -gt 1MB) { "{0:N1} MB" -f ($totalSize / 1MB) }
               elseif ($totalSize -gt 1KB) { "{0:N1} KB" -f ($totalSize / 1KB) }
               else { "$totalSize Bytes" }

    if ($DryRun) {
        Write-Item "$Label - 将删除 $($files.Count) 个文件 ($sizeStr)"
        $files | ForEach-Object { Write-Item "    $_" }
    } else {
        $files | Remove-Item -Force
        Write-Done "$Label - 已删除 $($files.Count) 个文件 ($sizeStr)"
    }
}

# ---- 参数校验：未指定任何类别时显示帮助 ----
if (-not ($All -or $Logs -or $Models -or $Replay -or $Metrics)) {
    Write-Host ""
    Write-Host "训练产物清理脚本" -ForegroundColor Cyan
    Write-Host "用法: .\clean.ps1 [-All] [-Logs] [-Models] [-Replay] [-Metrics] [-DryRun]" -ForegroundColor White
    Write-Host ""
    Write-Host "  -All       清理所有类别" -ForegroundColor White
    Write-Host "  -Logs      清理运行时日志（AIServer / TrainClient / Learner）" -ForegroundColor White
    Write-Host "  -Models    清理模型缓存（P2P ONNX + Checkpoint .pt）" -ForegroundColor White
    Write-Host "  -Replay    清理回放记录（TrainClient .jsonl 文件）" -ForegroundColor White
    Write-Host "  -Metrics   清理 Dashboard 指标数据（Learner .jsonl 指标）" -ForegroundColor White
    Write-Host "  -DryRun    预览模式，只显示不删除" -ForegroundColor White
    Write-Host ""
    Write-Host "示例:" -ForegroundColor Gray
    Write-Host "  .\clean.ps1 -All                  # 清理全部" -ForegroundColor Gray
    Write-Host "  .\clean.ps1 -Logs -Replay         # 只清理日志和回放" -ForegroundColor Gray
    Write-Host "  .\clean.ps1 -All -DryRun          # 预览全部清理目标" -ForegroundColor Gray
    exit 0
}

# ---- 开始清理 ----
$mode = if ($DryRun) { "预览模式" } else { "执行模式" }
Write-Host "`n[清理脚本] $mode - 项目根目录: $ProjectRoot" -ForegroundColor Magenta

# ---- 1. 日志清理 ----
if ($All -or $Logs) {
    Write-Header "日志清理"
    Remove-MatchedFiles -Directory "AIServer\log"      -Pattern "*.log"  -Label "AIServer 日志"
    Remove-MatchedFiles -Directory "TrainClient\log"    -Pattern "*.log"  -Label "TrainClient 日志"
    Remove-MatchedFiles -Directory "RL-Learner\logs"    -Pattern "*.log"  -Label "Learner 日志"
}

# ---- 2. 模型缓存清理 ----
if ($All -or $Models) {
    Write-Header "模型缓存清理"
    Remove-MatchedFiles -Directory "RL-Learner\models\p2p"  -Pattern "*.onnx" -Label "Learner P2P 模型"
    Remove-MatchedFiles -Directory "AIServer\models\p2p"    -Pattern "*.onnx" -Label "AIServer P2P 模型"
    Remove-MatchedFiles -Directory "RL-Learner\models\save" -Pattern "*.pt"   -Label "Learner Checkpoint"
}

# ---- 3. 回放文件清理 ----
if ($All -or $Replay) {
    Write-Header "回放文件清理"
    Remove-MatchedFiles -Directory "TrainClient\log\viz" -Pattern "*.jsonl" -Label "TrainClient 回放记录"
}

# ---- 4. 指标数据清理 ----
if ($All -or $Metrics) {
    Write-Header "指标数据清理"
    Remove-MatchedFiles -Directory "RL-Learner\logs\metrics" -Pattern "*.jsonl" -Label "Dashboard 指标数据"
}

# ---- 完成 ----
Write-Host ""
if ($DryRun) {
    Write-Host "[完成] 预览结束，未删除任何文件。去掉 -DryRun 参数执行实际清理。" -ForegroundColor Yellow
} else {
    Write-Host "[完成] 清理完毕。" -ForegroundColor Green
}
