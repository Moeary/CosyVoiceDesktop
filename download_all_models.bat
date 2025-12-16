@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ================================================
echo     CosyVoice 模型下载工具
echo     Model Download Tool
echo ================================================
echo.

REM Get the script directory
set "SCRIPT_DIR=%~dp0"

REM Try to find Python from pixi environment
set "PYTHON_EXE="

REM Method 1: Check .pixi\envs\default\Scripts\python.exe
if exist "!SCRIPT_DIR!.pixi\envs\default\Scripts\python.exe" (
    set "PYTHON_EXE=!SCRIPT_DIR!.pixi\envs\default\Scripts\python.exe"
    echo ✅ 找到 pixi 环境中的 Python
    goto :found_python
)

REM Method 2: Check .pixi\envs\default\python.exe
if exist "!SCRIPT_DIR!.pixi\envs\default\python.exe" (
    set "PYTHON_EXE=!SCRIPT_DIR!.pixi\envs\default\python.exe"
    echo ✅ 找到 pixi 环境中的 Python
    goto :found_python
)

REM Method 3: Find any python.exe in .pixi\envs\*\Scripts\
for /d %%D in ("!SCRIPT_DIR!.pixi\envs\*") do (
    if exist "%%D\Scripts\python.exe" (
        set "PYTHON_EXE=%%D\Scripts\python.exe"
        echo ✅ 找到 pixi 环境中的 Python: !PYTHON_EXE!
        goto :found_python
    )
)

REM Fallback to system Python
echo ⚠️ 未找到 pixi 环境，尝试使用系统 Python...
where python >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for /f "delims=" %%i in ('where python') do set "PYTHON_EXE=%%i"
    echo ⚠️ 使用系统 Python: !PYTHON_EXE!
    echo ⚠️ 警告：系统 Python 可能缺少必要的包（如 modelscope）
)

:found_python

REM If still not found, error
if "!PYTHON_EXE!"=="" (
    echo ❌ 未找到 Python，请确保已安装 pixi 或 Python
    echo ❌ Python not found, please install pixi or Python first
    pause
    exit /b 1
)

REM Run the download script
echo 🚀 启动下载脚本...
echo.

REM If no arguments, download all models
if "%1"=="" (
    "!PYTHON_EXE!" "!SCRIPT_DIR!core\download.py" --all
) else (
    "!PYTHON_EXE!" "!SCRIPT_DIR!core\download.py" %*
)

echo.
echo ================================================
echo 按任意键继续... / Press any key to continue...
echo ================================================
pause >nul
