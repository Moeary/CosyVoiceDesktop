# CosyVoice Desktop

面向多角色有声内容创作的桌面端 TTS 工具，基于 CosyVoice 构建，提供文本编辑、角色配置、批量生成和本地 API 服务。

详细教程、页面说明、排错和完整更新记录已迁移到 Wiki：
[https://github.com/Moeary/CosyVoiceDesktop/wiki](https://github.com/Moeary/CosyVoiceDesktop/wiki)

## 核心能力

- 多角色语音配置：角色名、参考文本、参考音频、模式、颜色统一管理。
- 文本分段标注：支持右键 / `Ctrl+数字` 手动打标签。
- AI 角色分配：调用 OpenAI 兼容 LLM 按段落自动识别角色，可手动确认或全自动应用。
- 本地 TTS API：兼容 SillyTavern，并新增 OpenAI TTS 风格接口。
- 模型下载与配置记忆：路径、项目、角色配置自动持久化。

## 快速开始

### 1. 下载与启动

- 发行版下载： [GitHub Releases](https://github.com/Moeary/CosyVoiceDesktop/releases)
- 项目主页： [Moeary/CosyVoiceDesktop](https://github.com/Moeary/CosyVoiceDesktop)

启动后按下面顺序使用：

1. 在“模型下载”页面准备 `wetext` 和 `CosyVoice3` 模型。
2. 在“语音设置”页面创建或导入角色配置。
3. 在“文本编辑”页面输入文本，手动标注或使用右侧角色控制台中的 `AI分配角色`。
4. 点击“`一键运行`”或转成任务计划批量生成。

## API 快速示例

### SillyTavern / 通用接口

获取角色列表：

```bash
curl http://127.0.0.1:9880/speakers
```

语音合成：

```bash
curl http://127.0.0.1:9880/ \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，欢迎使用 CosyVoice Desktop",
    "speaker": "旁白",
    "speed": 1.0
  }' \
  --output sample.wav
```

### OpenAI TTS 兼容接口

```bash
curl http://127.0.0.1:9880/v1/audio/speech \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini-tts",
    "input": "你好，欢迎使用 CosyVoice Desktop",
    "voice": "旁白",
    "response_format": "mp3"
  }' \
  --output sample.mp3
```

角色列表扩展端点：

```bash
curl http://127.0.0.1:9880/v1/audio/speakers
```

## 文档入口

- 新手教程： [Wiki Home](https://github.com/Moeary/CosyVoiceDesktop/wiki)
- 问题反馈： [Issues](https://github.com/Moeary/CosyVoiceDesktop/issues)

## 说明

- 详细安装步骤、模型下载说明、页面教程、FAQ、更新日志和示例素材说明已迁移到 Wiki。
- 请确保你对输入文本、参考音频和生成内容拥有合法使用权。
