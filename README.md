# CosyVoice Pro 有声小说生成器

面向创作者的多功能桌面端有声小说生产力工具，基于 CosyVoice 系列大模型构建，提供即开即用的文本转语音体验。

## � 项目简介

CosyVoice Pro 是在官方 CosyVoice 能力之上构建的图形化有声内容创作平台。通过 PyQt 与 Fluent Design 风格界面，整合零样本克隆、精细控制、指令式创作、流式生成等多种推理模式，帮助小说、广播剧、播客及教育内容创作者快速完成高质量的有声作品。

## ✨ 核心优势

- **一站式工作流**：从文本管理、角色配置到批量音频导出均在同一界面完成。
- **多场景语音模式**：零样本克隆、精细控制、指令控制、流式输入四种模式自由组合，覆盖旁白/人物/多语言需求。
- **色彩化标注体验**：不同角色以颜色区分，提升长篇文本的配音效率与可读性。
- **自动化播放与日志**：生成后自动按段播放并输出实时日志，快速定位问题。
- **配置复用**：支持语音配置保存/导入，多端协同创作毫不费力。

## 🔍 功能亮点

### 智能文本工作台
- 智能文本编辑器，支持复制/粘贴/剪切/撤销等常用操作。
- 按段落为文本绑定语音配置，实时查看配音颜色标记。
- 一键生成后自动顺序播放音频，历史记录清晰可追溯。

### 多语音角色管理
- 支持无限量语音配置，适配旁白、主要角色、群演等场景。
- 配置项包含模式、参考文本、参考音频、指令文本与标记颜色。
- 内置配置校验，缺失信息即时提示，避免生成失败。
- JSON 导入导出，可与团队成员共享同一套角色库。

### 创作辅助能力
- 适配中文、英文、日文、韩文及多种方言的跨语言创作需求。
- 流式输入模式可结合大语言模型逐句输出，实现长篇实时生成。
- 日志面板实时展示生成进度、耗时与潜在告警信息。
- 输出目录自动管理生成文件，支持批量回放与二次处理。

## 📸 界面示例

- ![主界面预览](链接待定)
- ![语音配置面板](链接待定)
- ![生成日志与播放器](链接待定)

## � 三种使用方式概览

| 方式 | 推荐人群 | 前置条件 | 快速操作 |
| --- | --- | --- | --- |
| 方式一：百度网盘一键包 | 想立即体验、拥有 NVIDIA GPU 的创作者 | 支持 CUDA ≥ 12.1 的 NVIDIA 显卡，Windows 10/11 | 下载压缩包 → 解压 → 双击运行 |
| 方式二：已有 CosyVoice 环境 | 已经本地部署官方 CosyVoice 的用户 | 本地 CosyVoice 目录与模型完整可用 | 安装桌面依赖 → 运行 `python cosyvoice_pro.py` |
| 方式三：全手动安装编译 | 希望了解完整部署流程的新手 | Git、Conda、Python 3.10、充足磁盘空间 | 从零克隆仓库 → 配置环境 → 拷贝 GUI → 编译/运行 |

> 🔔 提示：三种方式可并行维护，推荐保留同一套 `pretrained_models` 以节省磁盘空间。

## 方式一：百度网盘一键体验包（链接待定）

### 适用用户
- 需要最快上手体验 CosyVoice Pro 的创作者。
- 设备搭载 NVIDIA GPU，驱动已支持 CUDA 12.1 及以上版本。
- Windows 10/11 环境。

### 步骤
1. 访问百度网盘链接（待补充），下载最新发布的压缩包。
2. 在本地磁盘解压，例如 `D:\CosyVoicePro`。
3. 检查显卡驱动与 CUDA Runtime 是否满足 12.1 及以上要求。
4. 双击 `cosyvoice_pro.exe`（依据压缩包提供的启动脚本）。
5. 首次启动会自动校验依赖并加载模型，随后进入主界面即可使用。

### 注意事项
- 解压路径请避免中文或空格字符，以免影响 Python 虚拟环境。
- 若 Windows SmartScreen 拦截，可选择“更多信息 → 仍要运行”。
- 首次加载大型模型时耗时较长，耐心等待日志面板提示完成。

## 方式二：已有 CosyVoice 环境的快速集成

### 适用用户
- 已在本地 `CosyVoice` 源码目录中完成依赖安装与模型下载。
- 希望直接将 GUI 集成至现有环境，保持与官方脚本同一套虚拟环境。

### 操作步骤
1. 切换至 CosyVoice 根目录：
   ```powershell
   cd path\to\CosyVoice
   ```
2. 确保虚拟环境已激活且可正常运行官方脚本。
3. 安装桌面端依赖：
   ```powershell
   pip install "PyQt-Fluent-Widgets[full]" -i https://pypi.org/simple/
   ```
4. 将 `CosyVoiceGUI` 仓库中的 `cosyvoice_pro.py`、`config/`、`build_config.txt`、`build.ps1`（如需打包）复制到 CosyVoice 根目录或自定义工作目录：
   ```powershell
   copy path\to\CosyVoiceGUI\cosyvoice_pro.py  .
   robocopy path\to\CosyVoiceGUI\config .\config /E
   copy path\to\CosyVoiceGUI\build_config.txt .
   copy path\to\CosyVoiceGUI\build.ps1 .
   ```
5. 运行桌面应用：
   ```powershell
   python cosyvoice_pro.py
   ```
6. 首次运行建议在“语音设置”中校验模型路径与音频输出目录，确保指向现有的 `pretrained_models`。

### 可选：生成便携式版本
- 依据 `build_config.txt` 调整打包选项。
- 运行 `build.ps1` 以生成独立可执行程序，便于分发。

## 方式三：零基础全流程手动安装与编译

### 前置条件
- Windows 10/11。
- Git、Conda 或 Mambaforge、Python 3.10。
- 至少 30 GB 可用磁盘空间（含模型）。
- 可访问外网以下载依赖与模型，如需国内镜像可根据下述示例修改源。

### Step 1：获取官方 CosyVoice 源码
```powershell
cd D:\Projects
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
```

### Step 2：创建与配置 Python 环境
```powershell
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

若需语音前处理增强，可按需安装 `ttsfrd` 资源：
```powershell
cd pretrained_models\CosyVoice-ttsfrd
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-win_amd64.whl
```

### Step 3：下载预训练模型
可在 Python 交互环境中使用 ModelScope SDK：
```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

或使用 Git LFS：
```powershell
mkdir pretrained_models
git lfs install
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models\CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models\CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models\CosyVoice-ttsfrd
```

### Step 4：获取 CosyVoice Pro 桌面端资源
```powershell
cd ..
git clone https://github.com/Moeary/CosyVoiceGUI.git
cd CosyVoiceGUI
```

### Step 5：整合 GUI 与 CosyVoice 环境
- 将 `cosyvoice_pro.py` 放置于 `CosyVoice` 根目录或自定义的 `tools/gui` 目录。
- 同步 `config/`、`build_config.txt`、`build.ps1` 等文件至目标目录。
- 若希望统一管理输出目录，可在 `config/voice_config.json` 中调整 `output_path`。

示例（以 CosyVoice 根目录为目标）：
```powershell
copy cosyvoice_pro.py ..\CosyVoice
robocopy config ..\CosyVoice\config /E
copy build_config.txt ..\CosyVoice
copy build.ps1 ..\CosyVoice
```

### Step 6：安装桌面端依赖并验证运行
```powershell
conda activate cosyvoice
cd ..\CosyVoice
pip install "PyQt-Fluent-Widgets[full]" -i https://pypi.org/simple/
python cosyvoice_pro.py
```

如需生成独立发行版，可编辑 `build_config.txt` 后运行：
```powershell
pwsh build.ps1
```

### Step 7：首次使用建议
- 在“语音设置”中新增至少一个配置，选择零样本或精细控制模式。
- 导入对话文本，使用右键菜单给不同角色分配语音。
- 点击“生成语音”并留意日志输出，首个项目建议保存配置文件以方便复用。

## 常见问题 FAQ

1. **模型加载失败**：确认 `pretrained_models` 中的目录与配置指向一致，且显存足够（建议 ≥ 8 GB）。
2. **依赖安装报错**：优先检查 Python 版本为 3.10；若在国内网络，建议使用镜像源。
3. **界面空白或闪退**：确保已安装最新显卡驱动，必要时以管理员权限运行。
4. **音频无声或失真**：核对参考音频采样率（建议 16 kHz）与文本语言是否匹配。
5. **打包体积过大**：可在 `build_config.txt` 调整是否包含模型与虚拟环境。

## 更新与反馈

- 项目主页：链接待定
- 功能需求与问题反馈：请在 GitHub Issues 提交
- 企业合作或私有化部署：可邮件联系（地址待补充）

## 📝 用户协议

1. 本项目基于 CosyVoice 开源能力，遵循原项目许可证及使用规范。请在下载和部署前阅读并遵守 CosyVoice 官方条款。
2. 用户在创作过程中应确保拥有使用输入文本、参考音频及生成内容的合法权利，不得侵犯第三方的版权、肖像权或其他合法权益。
3. 禁止将本项目用于任何违法、违规或违背公共秩序与善良风俗的用途；如因违规使用导致损失，责任由用户自行承担。
4. 项目提供的打包版本及脚本仅供个人学习与研究使用，未经许可不得用于商业再发行或转售。
5. 项目维护者保留依据法律法规或社区反馈随时更新、暂停或终止服务与支持的权利。

> 使用 CosyVoice Pro 即视为同意上述协议条款。若您不同意任何条款，请立即停止使用并删除相关文件。
