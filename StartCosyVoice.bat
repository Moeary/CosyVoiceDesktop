@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title CosyVoice Desktop Pro

echo ====================================
echo    CosyVoice Desktop v1.3
echo ====================================
echo.

REM 获取当前脚本所在目录
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM 设置 Python 路径 - 优先使用 pixi 环境，回退到系统 Python
set "PYTHON="

REM Method 1: Check .pixi\envs\default\Scripts\python.exe
if exist "!SCRIPT_DIR!.pixi\envs\default\Scripts\python.exe" (
    set "PYTHON=!SCRIPT_DIR!.pixi\envs\default\Scripts\python.exe"
    echo ✅ 使用 pixi 环境中的 Python
    goto :run_program
)

REM Method 2: Check .pixi\envs\default\python.exe
if exist "!SCRIPT_DIR!.pixi\envs\default\python.exe" (
    set "PYTHON=!SCRIPT_DIR!.pixi\envs\default\python.exe"
    echo ✅ 使用 pixi 环境中的 Python
    goto :run_program
)

REM Method 3: Find any python.exe in .pixi\envs\*\Scripts\
for /d %%D in ("!SCRIPT_DIR!.pixi\envs\*") do (
    if exist "%%D\Scripts\python.exe" (
        set "PYTHON=%%D\Scripts\python.exe"
        echo ✅ 使用 pixi 环境中的 Python: !PYTHON!
        goto :run_program
    )
)

REM Fallback to system Python
echo ⚠️ 未找到 pixi 环境，尝试使用系统 Python...
for /f "delims=" %%i in ('where python 2^>nul') do (
    set "PYTHON=%%i"
    echo ✅ 使用系统 Python: !PYTHON!
    goto :run_program
)

:run_program

REM 检查 Python 是否存在
if "!PYTHON!"=="" (
    echo ❌ 错误：找不到 Python 环境。请安装 pixi 后运行 'pixi install' 或下载完整包。
    echo ❌ Error: Python environment not found. Please run 'pixi install' first or download the full package.
    pause
    exit /b 1
)

REM 运行主程序
"!PYTHON!" main.py %*

if errorlevel 1 (
    echo.
    echo ====================================
    echo [错误] 程序运行出错！
    echo ====================================
    echo.
    echo 可能原因:
    echo 1. 缺少必要的 DLL 文件
    echo 2. 模型文件不存在或路径错误
    echo 3. 显存不足或 CUDA 版本不匹配
    echo.
    echo 请查看上方错误信息以了解详情
    echo.
    pause
)
