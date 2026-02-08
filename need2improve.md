# Future Roadmap: Dockerization & Remote API Support

**目标**: 将 CosyVoice/Qwen3 推理服务容器化，并支持桌面端远程连接。

## 1. 架构调整 (Triple Mode)
最终目标是支持三种启动模式：
1.  **Embedded Mode**: 现有模式。GUI 自动拉起本地 API 子进程。
2.  **Client Mode**: 纯 GUI。不启动本地引擎，连接远程 IP (e.g., `http://192.168.1.50:9880`)。
3.  **Server Mode**: 纯 API。无 GUI 环境，通过 Docker 运行。

## 2. Docker 适配
### 依赖分离
- 当前 `pixi.toml` 混杂了 Windows 专用的 PyQt5 和 Torch Whl。
- **行动**: 创建 `requirements-server.txt`，仅包含服务端依赖 (torch-linux, fastapi, uvicorn, modelscope)，排除 `pyqt5`, `pyqt-fluent-widgets`。

### Dockerfile 示例
```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装系统依赖 (如有必要)
RUN apt-get update && apt-get install -y ffmpeg

# 复制依赖
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# 复制核心代码
COPY core/ /app/core/
COPY config/ /app/config/
COPY pretrained_models/ /app/pretrained_models/

# 暴露端口
EXPOSE 9880

# 启动命令 (需要修改 main.py 支持 --server 模式)
CMD ["python", "-m", "core.service.main", "--host", "0.0.0.0"]

3. 远程管理 API
为了让客户端能远程控制服务器，需要补充以下 API：

GET /admin/status: 查看显存、当前模型。
POST /admin/download: 远程触发模型下载。
GET /admin/download/progress: 轮询下载进度。
4. 遗留问题
模型文件挂载: Docker 运行时建议将 pretrained_models 映射为 Volume，避免每次重启重新下载。