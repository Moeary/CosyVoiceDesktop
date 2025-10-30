# CosyVoice Desktop Pro 启动脚本
$Host.UI.RawUI.WindowTitle = "CosyVoice Desktop"
Set-Location $PSScriptRoot

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "   CosyVoice Desktop v1.1" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[启动] 正在启动程序..." -ForegroundColor Yellow
Write-Host ""

$pythonExe = Join-Path $PSScriptRoot "python_env\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "[错误] Python 环境不存在！" -ForegroundColor Red
    Write-Host "请确保 python_env 目录完整" -ForegroundColor Yellow
    Read-Host "按回车键退出"
    exit 1
}

# 设置 PATH
$env:PATH = "$PSScriptRoot\python_env;$env:PATH"

& $pythonExe CosyVoiceDesktop.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "====================================" -ForegroundColor Red
    Write-Host "[错误] 程序运行出错！" -ForegroundColor Red
    Write-Host "====================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "按回车键退出"
}
