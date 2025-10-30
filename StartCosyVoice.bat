@echo off
chcp 65001 > nul
title CosyVoice Desktop Pro

echo ====================================
echo    CosyVoice Desktop v1.0.0
echo ====================================
echo.

cd /d "%~dp0"

echo [启动] 正在启动程序...
echo.

REM 设置 PATH，确保能找到所有 DLL
set "PATH=%~dp0python_env;%PATH%"

"%~dp0python_env\python.exe" CosyVoiceDesktop.py

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
