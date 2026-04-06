import argparse
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.modules.setdefault("core.api", sys.modules[__name__])
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "../third_party/AcademiCodec"))
sys.path.insert(0, os.path.join(ROOT_DIR, "../third_party/Matcha-TTS"))

api_logger = logging.getLogger("cosyvoice_api")
api_logger.setLevel(logging.INFO)
api_logger.propagate = False
for logger_name in ("cosyvoice", "Matcha-TTS", "uvicorn.error", "uvicorn.access", "httpx", "torch", "lightning"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

log_callbacks = []
MODE_LABELS = {"zero_shot": "语音克隆", "cross_lingual": "精细控制", "instruct": "指令模式"}
MODE_ALIASES = {
    "zero-shot": "zero_shot",
    "zero_shot": "zero_shot",
    "clone": "zero_shot",
    "clone_with_prompt": "zero_shot",
    "voice_clone": "zero_shot",
    "零样本复制": "zero_shot",
    "语音克隆": "zero_shot",
    "cross_lingual": "cross_lingual",
    "cross-lingual": "cross_lingual",
    "fine-grained": "cross_lingual",
    "fine_grained": "cross_lingual",
    "精细控制": "cross_lingual",
    "instruct": "instruct",
    "instruction": "instruct",
    "instruction_control": "instruct",
    "voice_design": "instruct",
    "指令控制": "instruct",
    "指令模式": "instruct",
}

cosyvoice = None
character_config = None
min_text_length = 0


class CallbackHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        for callback in log_callbacks:
            try:
                callback(msg)
            except Exception:
                pass


if not api_logger.handlers:
    callback_handler = CallbackHandler()
    callback_handler.setFormatter(logging.Formatter("%(message)s"))
    api_logger.addHandler(callback_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    api_logger.addHandler(console_handler)


def get_torch_modules():
    import torch
    import torchaudio

    return torch, torchaudio


def set_log_callback(callback):
    log_callbacks.clear()
    log_callbacks.append(callback)


def get_runtime_temp_dir() -> str:
    temp_dir = os.environ.get("COSYVOICE_TEMP_DIR", "").strip() or os.path.join(ROOT_DIR, "..", "runtime_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def create_named_temp_file(suffix: str):
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=get_runtime_temp_dir())


class CharacterConfig:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.characters = {}
        self.load_characters()

    def load_characters(self):
        if not os.path.exists(self.config_file):
            api_logger.warning(f"⚠️ Config file not found: {self.config_file}")
            return
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            if isinstance(config_data, list):
                for char in config_data:
                    char_name = char.get("name", "")
                    if char_name:
                        self.characters[char_name] = char
            else:
                self.characters[config_data.get("name", "default")] = config_data
            api_logger.info(f"✅ Loaded {len(self.characters)} characters from {os.path.basename(self.config_file)}")
        except Exception as exc:
            api_logger.error(f"❌ Failed to load {self.config_file}: {exc}")

    def get_character(self, char_name: str) -> dict | None:
        return self.characters.get(char_name)

    def list_characters(self) -> list[str]:
        return list(self.characters.keys())


def clean_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def prepare_request_text(text: str, *, field_name: str = "text") -> str:
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError(f"{field_name} 不能为空")
    if len(cleaned_text) < min_text_length:
        raise ValueError(f"文本长度({len(cleaned_text)}) < 最小长度({min_text_length}), 已跳过")
    return cleaned_text


def run_ffmpeg(input_file: str, output_file: str, args: list | None = None):
    cmd = ["ffmpeg", "-i", input_file, "-y"] + (args or []) + [output_file]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        api_logger.error(f"❌ FFmpeg error: {exc.stderr.decode(errors='ignore')}")
        return False
    except FileNotFoundError:
        api_logger.error("❌ FFmpeg not found in system PATH. Please install FFmpeg.")
        return False


def speed_change_ffmpeg(input_audio_path: str, speed: float, output_path: str) -> bool:
    if speed < 0.5 or speed > 2.0:
        speed = max(0.5, min(2.0, speed))
    return run_ffmpeg(input_audio_path, output_path, ["-filter:a", f"atempo={speed}"])


def load_cosyvoice_model():
    try:
        from .utils import load_cosyvoice_model as _load_model
    except ImportError:
        from core.utils import load_cosyvoice_model as _load_model
    return _load_model()


def set_globals(model, config_manager):
    global cosyvoice, character_config
    cosyvoice = model
    character_config = config_manager
    api_logger.info("✅ API globals set from external source")


app = FastAPI(
    title="CosyVoice API",
    version="1.0.0",
    description="SillyTavern, OpenAI-compatible, and native CosyVoice speech APIs.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def set_min_text_length(length: int):
    global min_text_length
    min_text_length = length


def extract_voice_name(voice_value) -> str:
    if isinstance(voice_value, dict):
        return str(voice_value.get("id", "") or voice_value.get("name", "")).strip()
    return str(voice_value or "").strip()


def normalize_mode_name(mode_value: str | None) -> str | None:
    mode_text = str(mode_value or "").strip()
    return MODE_ALIASES.get(mode_text, mode_text) if mode_text else None


def get_mode_label(mode_value: str | None) -> str:
    return MODE_LABELS.get(normalize_mode_name(mode_value), str(mode_value or ""))


def extract_request_field(payload: dict, *keys, default=None):
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return default


def save_uploaded_prompt_audio(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix or ".wav"
    with create_named_temp_file(suffix=suffix) as temp_audio:
        temp_audio.write(data)
        return temp_audio.name


def cleanup_temp_file(file_path: str | None):
    if file_path and os.path.exists(file_path):
        os.unlink(file_path)


def build_runtime_char_config(payload: dict, uploaded_prompt_audio_path: str | None = None):
    requested_character = str(
        extract_request_field(payload, "profile", "character_name", "speaker", "voice", default="") or ""
    ).strip()
    runtime_config = {}

    if requested_character:
        if character_config is None:
            raise ValueError("角色配置未加载，无法通过 profile/character_name/speaker/voice 查找默认配置")
        base_config = character_config.get_character(requested_character)
        if not base_config:
            raise ValueError(f"未找到角色: {requested_character}")
        runtime_config.update(dict(base_config))
        runtime_config.setdefault("name", requested_character)

    prompt_audio_path = str(
        extract_request_field(
            payload,
            "prompt_audio_path",
            "reference_audio_path",
            "ref_audio_path",
            default="",
        ) or ''
    ).strip()
    if uploaded_prompt_audio_path:
        prompt_audio_path = uploaded_prompt_audio_path

    prompt_text = str(
        extract_request_field(payload, "prompt_text", "reference_text", default="") or ""
    ).strip()
    instruct_text = str(
        extract_request_field(
            payload,
            "instruct_text",
            "instruction_text",
            "instructions",
            "voice_instruction",
            default="",
        ) or ""
    ).strip()
    prompt_lang = str(
        extract_request_field(payload, "prompt_lang", "reference_lang", default="") or ""
    ).strip()
    requested_mode = normalize_mode_name(extract_request_field(payload, "mode", "task_mode", "inference_mode"))
    base_mode = normalize_mode_name(runtime_config.get("mode"))

    if prompt_audio_path:
        runtime_config["prompt_audio"] = prompt_audio_path
    if prompt_text:
        runtime_config["prompt_text"] = prompt_text
    if instruct_text:
        runtime_config["instruct_text"] = instruct_text
    if prompt_lang:
        runtime_config["prompt_lang"] = prompt_lang

    if requested_mode:
        runtime_config["mode"] = requested_mode
    elif base_mode:
        runtime_config["mode"] = base_mode
    elif instruct_text and prompt_audio_path:
        runtime_config["mode"] = "instruct"
    elif prompt_text and prompt_audio_path:
        runtime_config["mode"] = "zero_shot"
    elif prompt_audio_path:
        runtime_config["mode"] = "cross_lingual"

    runtime_mode = normalize_mode_name(runtime_config.get("mode"))
    runtime_config["mode"] = runtime_mode

    if not runtime_mode:
        raise ValueError("缺少 mode。请显式传 mode，或提供 profile / prompt 条件")

    if runtime_mode == "zero_shot":
        if not runtime_config.get("prompt_audio"):
            raise ValueError("zero_shot 需要 prompt_audio_path 或上传 prompt_audio")
        if not runtime_config.get("prompt_text"):
            raise ValueError("zero_shot 需要 prompt_text")
    elif runtime_mode == "instruct":
        if not runtime_config.get("prompt_audio"):
            raise ValueError("instruct 需要 prompt_audio_path 或上传 prompt_audio")
        if not runtime_config.get("instruct_text"):
            raise ValueError("instruct 需要 instruct_text")
    elif runtime_mode == "cross_lingual":
        if not runtime_config.get("prompt_audio"):
            raise ValueError("cross_lingual 需要 prompt_audio_path 或上传 prompt_audio")
    else:
        raise ValueError(f"不支持的 mode: {runtime_mode}")

    prompt_audio = runtime_config.get("prompt_audio")
    if prompt_audio and not os.path.exists(prompt_audio):
        raise ValueError(f"参考音频不存在: {prompt_audio}")

    return runtime_config, requested_character


def build_native_cosyvoice_meta():
    return {
        "provider": "cosyvoice",
        "object": "capabilities",
        "paths": {"json": "/cosyvoice/speech", "multipart": "/cosyvoice/speech/upload"},
        "supports": {
            "openai_compatible": True,
            "native_json": True,
            "native_multipart": True,
            "prompt_audio_upload": True,
            "profile_lookup": character_config is not None,
            "token_passthrough": True,
        },
        "modes": [
            {
                "id": "zero_shot",
                "label": "语音克隆",
                "aliases": ["zero-shot", "clone", "clone_with_prompt", "零样本复制"],
                "required_fields": ["text", "prompt_audio_path|prompt_audio", "prompt_text"],
            },
            {
                "id": "cross_lingual",
                "label": "精细控制",
                "aliases": ["fine-grained", "fine_grained", "精细控制"],
                "required_fields": ["text", "prompt_audio_path|prompt_audio"],
            },
            {
                "id": "instruct",
                "label": "指令模式",
                "aliases": ["instruction", "instruction_control", "指令控制"],
                "required_fields": ["text", "prompt_audio_path|prompt_audio", "instruct_text"],
            },
        ],
        "response_formats": ["wav", "mp3", "flac", "aac", "opus", "pcm"],
        "fields": {
            "text": "必填，待合成文本，原样传递给后端 tokenizer，不做字符清洗",
            "profile": "可选，本地预设名称，先加载本地配置，再由请求字段覆盖",
            "mode": "可选，zero_shot | cross_lingual | instruct",
            "prompt_audio_path": "可选，服务端本地参考音频路径",
            "prompt_audio": "可选，multipart 文件上传字段",
            "prompt_text": "zero_shot 必填，参考音频对应文本",
            "instruct_text": "instruct 必填，音色/风格指令",
            "prompt_lang": "可选，预留字段",
            "speed": "可选，默认 1.0",
            "response_format": "可选，默认 wav",
        },
        "profiles": build_profile_items(),
    }


def build_profile_items() -> list:
    items = []
    if character_config is None:
        return items

    for char_name in character_config.list_characters():
        config = character_config.get_character(char_name) or {}
        mode = normalize_mode_name(config.get("mode")) or "zero_shot"
        items.append(
            {
                "id": char_name,
                "name": char_name,
                "mode": mode,
                "mode_label": get_mode_label(mode),
                "has_prompt_audio": bool(config.get("prompt_audio")),
                "has_prompt_text": bool(config.get("prompt_text")),
                "has_instruct_text": bool(config.get("instruct_text")),
            }
        )
    return items


def build_tavern_speakers() -> list:
    return [{"name": item["name"], "voice_id": item["id"]} for item in build_profile_items()]


def build_speaker_items() -> list:
    items = []
    for item in build_profile_items():
        items.append({"id": item["id"], "name": item["name"], "voice_id": item["id"], "object": "speaker"})
    return items


def build_model_items() -> list:
    return [
        {
            "id": "cosyvoice-openai-tts",
            "object": "model",
            "owned_by": "CosyVoiceDesktop",
        }
    ]


def convert_audio_buffer_format(audio_buffer: io.BytesIO, response_format: str):
    fmt = (response_format or 'wav').strip().lower()
    if fmt == 'wav':
        audio_buffer.seek(0)
        return audio_buffer, 'audio/wav'

    if fmt == 'pcm':
        torch, torchaudio = get_torch_modules()
        audio_buffer.seek(0)
        audio_data, _ = torchaudio.load(audio_buffer)
        pcm_data = audio_data.clamp(-1.0, 1.0).mul(32767).to(torch.int16)
        raw_buffer = io.BytesIO(pcm_data.transpose(0, 1).contiguous().numpy().tobytes())
        raw_buffer.seek(0)
        return raw_buffer, 'audio/pcm'

    ffmpeg_formats = {
        'mp3': ('audio/mpeg', '.mp3'),
        'flac': ('audio/flac', '.flac'),
        'aac': ('audio/aac', '.aac'),
        'opus': ('audio/ogg', '.opus'),
    }
    if fmt not in ffmpeg_formats:
        raise ValueError(f'不支持的输出格式: {fmt}')

    mime_type, suffix = ffmpeg_formats[fmt]
    with create_named_temp_file(suffix='.wav') as tmp_input:
        audio_buffer.seek(0)
        tmp_input.write(audio_buffer.read())
        temp_input_path = tmp_input.name

    with create_named_temp_file(suffix=suffix) as tmp_output:
        temp_output_path = tmp_output.name

    try:
        success = run_ffmpeg(temp_input_path, temp_output_path)
        if not success:
            raise RuntimeError(f'FFmpeg 转换 {fmt} 失败')

        with open(temp_output_path, 'rb') as f:
            output_buffer = io.BytesIO(f.read())
        output_buffer.seek(0)
        return output_buffer, mime_type
    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)


# ==================== Router registration ====================

try:
    from .api_routers import cosyvoice_native, openai_compat, system, tavern
except ImportError:
    from core.api_routers import cosyvoice_native, openai_compat, system, tavern

for router_module in (system, tavern, openai_compat, cosyvoice_native):
    app.include_router(router_module.router)

# ==================== 推理核心逻辑 ====================

def _inference(text: str, char_config: dict, mode: str | None = None, speed: float = 1.0):
    start_time = time.time()
    try:
        torch, torchaudio = get_torch_modules()
        if cosyvoice is None:
            api_logger.error("Model not loaded")
            return None

        display_text = text[:100] + "..." if len(text) > 100 else text
        api_logger.info(f"📝 推理文本: {display_text}")

        resolved_mode = normalize_mode_name(mode or char_config.get("mode") or "zero_shot")
        if not resolved_mode:
            api_logger.error("❌ Missing inference mode")
            return None

        tts_speeches = []

        if resolved_mode == "zero_shot":
            prompt_audio_path = char_config.get("prompt_audio")
            prompt_text = char_config.get("prompt_text")
            if not prompt_audio_path or not os.path.exists(prompt_audio_path):
                api_logger.error(f"❌ [语音克隆] Prompt audio not found: {prompt_audio_path}")
                return None
            if not prompt_text:
                api_logger.error("❌ [语音克隆] Prompt text not found in config")
                return None

            is_v3 = "CosyVoice3" in getattr(cosyvoice, "model_dir", "")
            if is_v3 and "<|endofprompt|>" not in prompt_text:
                prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"

            try:
                for output in cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio_path):
                    tts_speeches.append(output["tts_speech"])
            except Exception as e:
                api_logger.error(f"❌ [语音克隆] 推理异常: {e}")
                import traceback

                traceback.print_exc()
                return None

        elif resolved_mode == "instruct":
            prompt_audio_path = char_config.get("prompt_audio")
            instruct_text = char_config.get("instruct_text", "")
            if not prompt_audio_path or not os.path.exists(prompt_audio_path):
                api_logger.error(f"❌ [指令模式] Prompt audio not found: {prompt_audio_path}")
                return None
            if not instruct_text:
                api_logger.error("❌ [指令模式] Instruction text not found in config")
                return None

            is_v3 = "CosyVoice3" in getattr(cosyvoice, "model_dir", "")
            if is_v3:
                if "<|endofprompt|>" not in instruct_text:
                    instruct_text = f"{instruct_text}<|endofprompt|>"
                if "You are a helpful assistant." not in instruct_text:
                    instruct_text = f"You are a helpful assistant. {instruct_text}"

            try:
                for output in cosyvoice.inference_instruct2(text, instruct_text, prompt_audio_path):
                    tts_speeches.append(output["tts_speech"])
            except Exception as e:
                api_logger.error(f"❌ [指令模式] 推理异常: {e}")
                import traceback

                traceback.print_exc()
                return None

        elif resolved_mode == "cross_lingual":
            prompt_audio_path = char_config.get("prompt_audio")
            if not prompt_audio_path or not os.path.exists(prompt_audio_path):
                api_logger.error(f"❌ [精细控制] Prompt audio not found: {prompt_audio_path}")
                return None

            tts_text = text
            is_v3 = "CosyVoice3" in getattr(cosyvoice, "model_dir", "")
            if is_v3 and "<|endofprompt|>" not in tts_text:
                tts_text = f"You are a helpful assistant.<|endofprompt|>{tts_text}"

            try:
                for output in cosyvoice.inference_cross_lingual(tts_text, prompt_audio_path):
                    tts_speeches.append(output["tts_speech"])
            except Exception as e:
                api_logger.error(f"❌ [精细控制] 推理异常: {e}")
                import traceback

                traceback.print_exc()
                return None

        else:
            api_logger.error(f"❌ Unknown mode: {resolved_mode}")
            return None

        if not tts_speeches:
            return None

        audio_data = torch.concat(tts_speeches, dim=1)

        if speed != 1.0:
            sample_rate = getattr(cosyvoice, "sample_rate", 22050)
            with create_named_temp_file(suffix=".wav") as tmp_input:
                torchaudio.save(tmp_input.name, audio_data, sample_rate, format="wav")
                temp_input_path = tmp_input.name

            with create_named_temp_file(suffix=".wav") as tmp_output:
                temp_output_path = tmp_output.name

            if speed_change_ffmpeg(temp_input_path, speed, temp_output_path):
                audio_data, _ = torchaudio.load(temp_output_path)
            else:
                api_logger.warning("⚠️ Speed change failed, returning original audio")

            cleanup_temp_file(temp_input_path)
            cleanup_temp_file(temp_output_path)

        buffer = io.BytesIO()
        sample_rate = getattr(cosyvoice, "sample_rate", 22050)
        torchaudio.save(buffer, audio_data, sample_rate, format="wav")
        buffer.seek(0)

        audio_duration = audio_data.shape[1] / sample_rate if audio_data.numel() > 0 else 0
        audio_size_mb = buffer.getbuffer().nbytes / (1024 * 1024)

        total_time = time.time() - start_time
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        api_logger.info(
            "✅ 推理成功 | ⏱️ 耗时: %.2fs | ⚡ RTF: %.4f | 🎵 时长: %.2fs | 💾 大小: %.2fMB",
            total_time,
            rtf,
            audio_duration,
            audio_size_mb,
        )
        return buffer

    except Exception as e:
        api_logger.error(f"❌ [推理] 总体异常: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='CosyVoice3 API Server')
    parser.add_argument(
        '--config',
        type=str,
        default='config/角色.json',
        help='角色配置文件路径 (默认: config/角色.json)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器地址 (默认: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9880,
        help='服务器端口 (默认: 9880)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    parser.add_argument(
        '--min_text_length',
        type=int,
        default=0,
        help='设置最小文本长度，低于该长度的请求将被跳过 (默认: 0，不限制)'
    )
    args = parser.parse_args()

    config_file = args.config
    if not os.path.isabs(config_file):
        config_file = os.path.join(ROOT_DIR, '..', config_file)

    if not config_file.endswith('.json'):
        print(f"❌ Error: --config must point to a .json file, got: {config_file}")
        sys.exit(1)

    global character_config, cosyvoice
    character_config = CharacterConfig(config_file)
    set_min_text_length(args.min_text_length)

    api_logger.info(f"📦 Loading CosyVoice model...")
    try:
        cosyvoice = load_cosyvoice_model()
        api_logger.info(f"✅ Model loaded successfully")
    except Exception as e:
        api_logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n🚀 Starting CosyVoice API Server...")
    print(f"📍 Host: {args.host}:{args.port}")
    print(f"🔗 Health check: http://{args.host}:{args.port}/health")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level='info' if args.debug else 'warning',
        access_log=False,
    )


if __name__ == "__main__":
    main()
