import time
import io
import os
import sys
import json
import argparse
import subprocess
import tempfile
import logging
import warnings
from pathlib import Path

# 禁用 tqdm
os.environ["TQDM_DISABLE"] = "1"
# 忽略所有警告
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, '../third_party/AcademiCodec'))
sys.path.insert(0, os.path.join(ROOT_DIR, '../third_party/Matcha-TTS'))

import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS

# ==================== 日志配置 ====================

# 创建 logger
api_logger = logging.getLogger('cosyvoice_api')
api_logger.setLevel(logging.INFO)
api_logger.propagate = False  # 防止日志传播到根记录器导致重复

# 屏蔽第三方库的繁杂日志
logging.getLogger('cosyvoice').setLevel(logging.ERROR)
logging.getLogger('Matcha-TTS').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('lightning').setLevel(logging.ERROR)

# 创建一个处理器用于日志回调
log_callbacks = []


def get_torch_modules():
    import torch
    import torchaudio
    return torch, torchaudio

class CallbackHandler(logging.Handler):
    """日志处理器，调用回调函数"""
    def emit(self, record):
        msg = self.format(record)
        for callback in log_callbacks:
            try:
                callback(msg)
            except:
                pass

# 添加处理器
callback_handler = CallbackHandler()
callback_handler.setFormatter(logging.Formatter('%(message)s'))
api_logger.addHandler(callback_handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
api_logger.addHandler(console_handler)

def set_log_callback(callback):
    """设置日志回调"""
    # 清空之前的回调，防止重启后日志重复输出
    log_callbacks.clear()
    log_callbacks.append(callback)

# ==================== 配置加载 ====================

class CharacterConfig:
    """角色配置管理"""
    
    def __init__(self, config_file: str):
        """
        初始化角色配置，读取单个 JSON 文件
        
        Args:
            config_file: 配置文件路径（必须是单个 .json 文件）
        """
        self.config_file = config_file
        self.characters = {}
        self.load_characters()
    
    def load_characters(self):
        """从 JSON 文件加载角色配置"""
        if not os.path.exists(self.config_file):
            api_logger.warning(f"⚠️ Config file not found: {self.config_file}")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 支持两种格式: 数组或单个对象
            if isinstance(config_data, list):
                # 数组格式: [{name, mode, prompt_text, prompt_audio, instruct_text, color}, ...]
                for char in config_data:
                    char_name = char.get('name', '')
                    if char_name:
                        self.characters[char_name] = char
            else:
                # 单个对象格式
                char_name = config_data.get('name', 'default')
                self.characters[char_name] = config_data
            
            api_logger.info(f"✅ Loaded {len(self.characters)} characters from {os.path.basename(self.config_file)}")
        
        except Exception as e:
            api_logger.error(f"❌ Failed to load {self.config_file}: {e}")
    
    def get_character(self, char_name: str) -> dict:
        """获取角色配置"""
        return self.characters.get(char_name, None)
    
    def list_characters(self) -> list:
        """列出所有角色"""
        return list(self.characters.keys())


# ==================== 文本处理 ====================

def clean_text(text: str) -> str:
    """
    清理和规范化文本
    移除可能导致问题的特殊字符，但保留中英文和常用标点
    """
    if not text:
        return text
    
    # 保留的字符集
    allowed_chars = set()
    # 中文
    for i in range(0x4E00, 0x9FFF + 1):
        allowed_chars.add(chr(i))
    # 英文字母
    for i in range(ord('a'), ord('z') + 1):
        allowed_chars.add(chr(i))
    for i in range(ord('A'), ord('Z') + 1):
        allowed_chars.add(chr(i))
    # 数字
    for i in range(ord('0'), ord('9') + 1):
        allowed_chars.add(chr(i))
    # 常用标点和空白符
    allowed_chars.update([' ', '，', '。', '！', '？', '；', '：', '"', '"', ''', ''', '、', 
                         ',', '.', '!', '?', ';', ':', '"', "'", '-', '—', '～', '…',
                         '\n', '\t', '(', ')', '（', '）', '[', ']', '【', '】', '{', '}'])
    
    # 清理文本
    cleaned = ''.join(c for c in text if c in allowed_chars or ord(c) > 127 and c not in '\x00\x01\x02')
    
    # 移除多个连续空白符
    while '  ' in cleaned:
        cleaned = cleaned.replace('  ', ' ')
    
    return cleaned.strip()


# ==================== FFmpeg 处理 ====================

def run_ffmpeg(input_file: str, output_file: str, args: list = None):
    """
    使用系统 FFmpeg 进行转换
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        args: 额外的 FFmpeg 参数
    """
    if args is None:
        args = []
    
    cmd = ['ffmpeg', '-i', input_file, '-y'] + args + [output_file]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found in system PATH. Please install FFmpeg.")
        return False


def speed_change_ffmpeg(input_audio_path: str, speed: float, output_path: str) -> bool:
    """
    使用 FFmpeg 进行变速处理
    
    Args:
        input_audio_path: 输入音频文件
        speed: 速度倍数 (0.5-2.0)
        output_path: 输出文件路径
    
    Returns:
        成功返回 True，失败返回 False
    """
    if speed < 0.5 or speed > 2.0:
        print(f"⚠️ Speed out of range [0.5-2.0], clamping to valid range")
        speed = max(0.5, min(2.0, speed))
    
    # FFmpeg atempo 滤镜用于变速
    filter_args = ['-filter:a', f'atempo={speed}']
    return run_ffmpeg(input_audio_path, output_path, filter_args)


# ==================== CosyVoice3 初始化 ====================

def load_cosyvoice_model():
    """加载本地CosyVoice模型 - 由 utils 模块负责"""
    try:
        # 尝试相对导入（当作为包导入时）
        from .utils import load_cosyvoice_model as _load_model
    except ImportError:
        # 当作为主脚本运行时，使用绝对导入
        from core.utils import load_cosyvoice_model as _load_model
    return _load_model()

# 全局模型实例
cosyvoice = None

# ==================== 依赖注入 ====================

def set_globals(model, config_manager):
    """
    设置全局变量，用于从 GUI 调用
    
    Args:
        model: CosyVoice 模型实例
        config_manager: 角色配置管理器 (需要实现 get_character 和 list_characters)
    """
    global cosyvoice, character_config
    cosyvoice = model
    character_config = config_manager
    print("✅ API globals set from external source")

# ==================== Flask 应用 ====================

app = Flask(__name__)
# 修复 CORS 参数
CORS(app)

# 全局角色配置（在启动时初始化）
character_config = None

# 全局最小文本长度配置
min_text_length = 0  # 默认0个字符，不限制

def set_min_text_length(length: int):
    """设置最小文本长度"""
    global min_text_length
    min_text_length = length
    api_logger.debug(f'✅ Min text length set to {length}')


def make_json_response(payload: dict | list, status: int = 200):
    response = app.response_class(
        response=json.dumps(payload, ensure_ascii=False),
        status=status,
        mimetype='application/json'
    )
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def make_options_response():
    response = app.response_class(
        response='',
        status=200,
        mimetype='text/plain'
    )
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


def extract_voice_name(voice_value) -> str:
    if isinstance(voice_value, dict):
        return str(voice_value.get('id', '') or voice_value.get('name', '')).strip()
    return str(voice_value or '').strip()


def build_speaker_items() -> list:
    items = []
    if character_config is None:
        return items

    for char_name in character_config.list_characters():
        items.append({
            'id': char_name,
            'name': char_name,
            'voice_id': char_name,
            'object': 'speaker'
        })
    return items


def build_model_items() -> list:
    return [
        {
            'id': 'cosyvoice-openai-tts',
            'object': 'model',
            'owned_by': 'CosyVoiceDesktop',
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
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
        audio_buffer.seek(0)
        tmp_input.write(audio_buffer.read())
        temp_input_path = tmp_input.name

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_output:
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


# ==================== 酒馆标准 API 端点 ====================

# ==================== 酒馆标准 API 端点 ====================

@app.route('/', methods=['POST', 'OPTIONS'])
def tts_tavern():
    """
    酒馆 TTS 根路由
    支持 POST 和 OPTIONS (CORS 预检)
    
    请求格式:
    {
        "text": "要生成的文本",
        "speaker": "角色名称 (voice_id)",
        "speed": 1.0 (可选)
    }
    """
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        response = app.response_class(
            response='',
            status=200,
            mimetype='text/plain'
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    try:
        if cosyvoice is None:
            response = app.response_class(
                response=json.dumps({'error': '模型未加载'}),
                status=500,
                mimetype='application/json'
            )
            return response
        
        data = request.get_json()
        
        text = data.get('text', '').strip()
        character_name = data.get('speaker', '').strip()  # 酒馆使用 speaker 参数
        speed = float(data.get('speed', 1.0))
        
        api_logger.info(f"📝 POST / request: speaker={character_name}, speed={speed}, text_len={len(text)}")
        
        if not text:
            response = app.response_class(
                response=json.dumps({'error': '文本不能为空'}),
                status=400,
                mimetype='application/json'
            )
            return response
        
        if not character_name:
            response = app.response_class(
                response=json.dumps({'error': '角色名不能为空'}),
                status=400,
                mimetype='application/json'
            )
            return response
        
        # 获取角色配置
        char_config = character_config.get_character(character_name)
        if not char_config:
            response = app.response_class(
                response=json.dumps({'error': f'未找到角色: {character_name}'}),
                status=404,
                mimetype='application/json'
            )
            return response
        
        # 清理文本
        original_text = text
        text = clean_text(text)
        if len(text) != len(original_text):
            api_logger.warning(f'Text cleaned: {len(original_text)} -> {len(text)} chars')
        
        # 检查最小长度
        if len(text) < min_text_length:
            error_msg = f'文本长度({len(text)}) < 最小长度({min_text_length}), 已跳过'
            api_logger.warning(error_msg)
            response = app.response_class(
                response=json.dumps({'error': error_msg}),
                status=400,
                mimetype='application/json'
            )
            return response
        
        api_logger.info(f'🎯 开始推理: 模式={char_config.get("mode")}, 速度={speed}, 文本长度={len(text)}')
        
        # 调用推理（不指定模式，从配置读取）
        audio_buffer = _inference(
            text=text,
            char_config=char_config,
            mode=None,
            speed=speed
        )
        
        if audio_buffer is None:
            error_msg = f'生成音频失败 (text_len={len(text)}, mode={char_config.get("mode")})'
            api_logger.error(error_msg)
            response = app.response_class(
                response=json.dumps({'error': error_msg}),
                status=500,
                mimetype='application/json'
            )
            return response
        
        audio_buffer.seek(0)
        return Response(audio_buffer.read(), mimetype='audio/wav')
    
    except Exception as e:
        print(f"❌ Error in POST /: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f'请求异常: {str(e)[:100]}'
        response = app.response_class(
            response=json.dumps({'error': error_msg}),
            status=500,
            mimetype='application/json'
        )
        return response

@app.route('/api/tts', methods=['POST', 'OPTIONS'])
def tts_api():
    """
    标准 TTS API 端点
    
    请求格式:
    {
        "text": "要生成的文本",
        "character_name": "角色名称",
        "mode": "零样本复制|精细控制|指令控制",
        "speed": 1.0
    }
    """
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        response = app.response_class(
            response='',
            status=200,
            mimetype='text/plain'
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    try:
        if cosyvoice is None:
            response = app.response_class(
                response=json.dumps({'error': '模型未加载，请检查模型文件'}),
                status=500,
                mimetype='application/json'
            )
            return response
        
        data = request.get_json()
        
        text = data.get('text', '').strip()
        character_name = data.get('character_name', '').strip()
        mode = data.get('mode', None)
        speed = float(data.get('speed', 1.0))
        
        api_logger.info(f"📝 POST /api/tts: character={character_name}, mode={mode}, speed={speed}, text_len={len(text)}")
        
        if not text:
            response = app.response_class(
                response=json.dumps({'error': '文本不能为空'}),
                status=400,
                mimetype='application/json'
            )
            return response
        
        if not character_name:
            response = app.response_class(
                response=json.dumps({'error': '角色名不能为空'}),
                status=400,
                mimetype='application/json'
            )
            return response
        
        # 获取角色配置
        char_config = character_config.get_character(character_name)
        if not char_config:
            response = app.response_class(
                response=json.dumps({'error': f'未找到角色: {character_name}'}),
                status=404,
                mimetype='application/json'
            )
            return response
        
        # 清理文本
        original_text = text
        text = clean_text(text)
        if len(text) != len(original_text):
            api_logger.warning(f'Text cleaned: {len(original_text)} -> {len(text)} chars')
        
        # 检查最小长度
        if len(text) < min_text_length:
            error_msg = f'文本长度({len(text)}) < 最小长度({min_text_length}), 已跳过'
            api_logger.warning(error_msg)
            response = app.response_class(
                response=json.dumps({'error': error_msg}),
                status=400,
                mimetype='application/json'
            )
            return response
        
        api_logger.info(f'🎯 开始推理: 模式={mode or char_config.get("mode")}, 速度={speed}, 文本长度={len(text)}')
        
        # 调用推理
        audio_buffer = _inference(
            text=text,
            char_config=char_config,
            mode=mode,
            speed=speed
        )
        
        if audio_buffer is None:
            error_msg = f'生成音频失败 (text_len={len(text)}, mode={mode or char_config.get("mode")})'
            api_logger.error(error_msg)
            response = app.response_class(
                response=json.dumps({'error': error_msg}),
                status=500,
                mimetype='application/json'
            )
            return response
        
        audio_buffer.seek(0)
        return Response(audio_buffer.read(), mimetype='audio/wav')
    
    except Exception as e:
        print(f"❌ Error in /api/tts: {e}")
        import traceback
        traceback.print_exc()
        error_msg = f'请求异常: {str(e)[:100]}'
        response = app.response_class(
            response=json.dumps({'error': error_msg}),
            status=500,
            mimetype='application/json'
        )
        return response

@app.route('/api/characters', methods=['GET'])
def list_characters():
    """
    获取所有可用角色列表
    只返回 name 和 voice_id，不暴露内部配置（prompt_text 等）
    """
    try:
        characters = []
        for char_name in character_config.list_characters():
            # 只返回 name 和 voice_id，不读取完整配置
            characters.append({
                'name': char_name,
                'voice_id': char_name
            })
        
        response = app.response_class(
            response=json.dumps(characters),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        print(f"❌ Error in /api/characters: {e}")
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        return response

@app.route('/speakers', methods=['GET', 'OPTIONS'])
def get_speakers():
    """
    酒馆 API 兼容端点 - 获取角色列表
    返回标准 JSON 数组格式: [{name, voice_id}, ...]
    不返回 prompt_text (这是内部配置，不暴露给酒馆)
    """
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        response = app.response_class(
            response='',
            status=200,
            mimetype='text/plain'
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    try:
        speakers = []
        for char_name in character_config.list_characters():
            # 只返回 name 和 voice_id，不返回 prompt_text
            speakers.append({
                'name': char_name,
                'voice_id': char_name
            })
        
        # 使用 app.response_class 确保返回原始 JSON 数组
        response = app.response_class(
            response=json.dumps(speakers),
            status=200,
            mimetype='application/json'
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        print(f"❌ Error in /speakers: {e}")
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

@app.route('/v1/audio/speakers', methods=['GET', 'OPTIONS'])
@app.route('/v1/audio/voices', methods=['GET', 'OPTIONS'])
def get_openai_speakers():
    """
    OpenAI 兼容扩展端点 - 返回本地配置中的角色列表
    """
    if request.method == 'OPTIONS':
        return make_options_response()

    try:
        return make_json_response({
            'object': 'list',
            'data': build_speaker_items()
        })
    except Exception as e:
        return make_json_response({'error': str(e)}, status=500)


@app.route('/v1/models', methods=['GET', 'OPTIONS'])
def get_openai_models():
    """
    OpenAI 兼容模型列表端点
    """
    if request.method == 'OPTIONS':
        return make_options_response()

    try:
        return make_json_response({
            'object': 'list',
            'data': build_model_items()
        })
    except Exception as e:
        return make_json_response({'error': str(e)}, status=500)


@app.route('/v1/audio/speech', methods=['POST', 'OPTIONS'])
def openai_audio_speech():
    """
    OpenAI TTS 兼容端点

    请求格式:
    {
        "model": "gpt-4o-mini-tts",
        "input": "要生成的文本",
        "voice": "角色名称" 或 {"id": "角色名称"},
        "instructions": "附加指令",
        "speed": 1.0,
        "response_format": "mp3|wav|flac|aac|opus|pcm"
    }
    """
    if request.method == 'OPTIONS':
        return make_options_response()

    try:
        if cosyvoice is None:
            return make_json_response({'error': '模型未加载'}, status=500)
        if character_config is None:
            return make_json_response({'error': '角色配置未加载'}, status=500)

        data = request.get_json(silent=True) or {}
        text = str(data.get('input', '') or data.get('text', '')).strip()
        voice_name = extract_voice_name(data.get('voice', '') or data.get('speaker', ''))
        speed = float(data.get('speed', 1.0))
        response_format = str(data.get('response_format', '') or data.get('format', '') or 'mp3').strip().lower()
        instructions = str(data.get('instructions', '') or '').strip()
        override_mode = str(data.get('mode', '') or '').strip() or None

        api_logger.info(
            f"📝 POST /v1/audio/speech: voice={voice_name}, format={response_format}, "
            f"speed={speed}, text_len={len(text)}"
        )

        if not text:
            return make_json_response({'error': 'input 不能为空'}, status=400)
        if not voice_name:
            return make_json_response({'error': 'voice 不能为空'}, status=400)

        char_config = character_config.get_character(voice_name)
        if not char_config:
            return make_json_response({'error': f'未找到角色: {voice_name}'}, status=404)

        char_config = dict(char_config)
        if instructions:
            char_config['instruct_text'] = instructions
            char_config['mode'] = '指令控制'

        original_text = text
        text = clean_text(text)
        if len(text) != len(original_text):
            api_logger.warning(f'Text cleaned: {len(original_text)} -> {len(text)} chars')

        if len(text) < min_text_length:
            error_msg = f'文本长度({len(text)}) < 最小长度({min_text_length}), 已跳过'
            api_logger.warning(error_msg)
            return make_json_response({'error': error_msg}, status=400)

        audio_buffer = _inference(
            text=text,
            char_config=char_config,
            mode=override_mode,
            speed=speed
        )
        if audio_buffer is None:
            return make_json_response({'error': '生成音频失败'}, status=500)

        converted_buffer, mime_type = convert_audio_buffer_format(audio_buffer, response_format)
        converted_buffer.seek(0)
        response = Response(converted_buffer.read(), mimetype=mime_type)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Content-Disposition'] = f'inline; filename=\"speech.{response_format}\"'
        return response

    except ValueError as e:
        return make_json_response({'error': str(e)}, status=400)
    except Exception as e:
        print(f"❌ Error in /v1/audio/speech: {e}")
        import traceback
        traceback.print_exc()
        return make_json_response({'error': f'请求异常: {str(e)[:150]}'}, status=500)

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    try:
        response = app.response_class(
            response=json.dumps({
                'status': 'ok',
                'model': 'CosyVoice3-0.5B',
                'characters': character_config.list_characters()
            }),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'status': 'error', 'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        return response

# ==================== 推理核心逻辑 ====================

def _inference(text: str, char_config: dict, mode: str = None, speed: float = 1.0):
    """
    CosyVoice3 推理
    
    Args:
        text: 要生成的文本
        char_config: 角色配置
        mode: 推理模式 (零样本复制|精细控制|指令控制)
        speed: 语速倍数
    
    Returns:
        包含WAV数据的BytesIO对象
    """
    start_time = time.time()
    try:
        torch, torchaudio = get_torch_modules()
        if cosyvoice is None:
            api_logger.error("Model not loaded")
            return None
        
        # 显示推理文本（截断显示前100个字符）
        display_text = text[:100] + "..." if len(text) > 100 else text
        api_logger.info(f"📝 推理文本: {display_text}")
        
        if mode is None:
            mode = char_config.get('mode', '零样本复制')
        
        # 规范化模式名称（如果是英文别名）
        mode_mapping = {
            'zero-shot': '零样本复制',
            'fine-grained': '精细控制',
            'instruction': '指令控制',
        }
        mode = mode_mapping.get(mode, mode)
        
        tts_speeches = []
        
        if mode == '零样本复制':
            # 零样本模式：使用参考音频和文本
            prompt_audio_path = char_config.get('prompt_audio')
            prompt_text = char_config.get('prompt_text')
            
            if not prompt_audio_path or not os.path.exists(prompt_audio_path):
                print(f"❌ [零样本] Prompt audio not found: {prompt_audio_path}")
                return None
            
            if not prompt_text:
                api_logger.error(f"❌ [零样本] Prompt text not found in config")
                return None
            
            api_logger.debug(f"[零样本] 参考音频: {os.path.basename(prompt_audio_path)}")
            
            # CosyVoice3 需要特定的 prompt 格式
            is_v3 = 'CosyVoice3' in getattr(cosyvoice, 'model_dir', '')
            if is_v3 and '<|endofprompt|>' not in prompt_text:
                prompt_text = f'You are a helpful assistant.<|endofprompt|>{prompt_text}'
            
            try:
                # 直接传递路径，让 CosyVoice 内部处理音频加载
                for output in cosyvoice.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_audio_path
                ):
                    tts_speeches.append(output['tts_speech'])
            except Exception as e:
                api_logger.error(f"❌ [零样本] 推理异常: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        elif mode == '指令控制':
            # 指令模式：通过指令控制音色
            prompt_audio_path = char_config.get('prompt_audio')
            instruct_text = char_config.get('instruct_text', '')
            
            if not prompt_audio_path or not os.path.exists(prompt_audio_path):
                api_logger.error(f"❌ [指令控制] Prompt audio not found: {prompt_audio_path}")
                return None
            
            if not instruct_text:
                api_logger.error(f"❌ [指令控制] Instruction text not found in config")
                return None
            
            api_logger.debug(f"[指令控制] 参考音频: {os.path.basename(prompt_audio_path)}")
            
            # CosyVoice3 指令模式使用 inference_instruct2
            is_v3 = 'CosyVoice3' in getattr(cosyvoice, 'model_dir', '')
            if is_v3:
                if '<|endofprompt|>' not in instruct_text:
                    instruct_text = f'{instruct_text}<|endofprompt|>'
                if 'You are a helpful assistant.' not in instruct_text:
                    instruct_text = f'You are a helpful assistant. {instruct_text}'
            
            try:
                for output in cosyvoice.inference_instruct2(
                    text,
                    instruct_text,
                    prompt_audio_path
                ):
                    tts_speeches.append(output['tts_speech'])
            except Exception as e:
                api_logger.error(f"❌ [指令控制] 推理异常: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        elif mode == '精细控制':
            # 精细控制模式：通过参考音频控制
            prompt_audio_path = char_config.get('prompt_audio')
            
            if not prompt_audio_path or not os.path.exists(prompt_audio_path):
                api_logger.error(f"❌ [精细控制] Prompt audio not found: {prompt_audio_path}")
                return None
            
            api_logger.debug(f"[精细控制] 参考音频: {os.path.basename(prompt_audio_path)}")
            
            # CosyVoice3 精细控制需要在文本前加指令
            tts_text = text
            is_v3 = 'CosyVoice3' in getattr(cosyvoice, 'model_dir', '')
            if is_v3 and '<|endofprompt|>' not in tts_text:
                tts_text = f'You are a helpful assistant.<|endofprompt|>{tts_text}'
            
            try:
                # 使用 inference_cross_lingual
                for output in cosyvoice.inference_cross_lingual(
                    tts_text,
                    prompt_audio_path
                ):
                    tts_speeches.append(output['tts_speech'])
            except Exception as e:
                api_logger.error(f"❌ [精细控制] 推理异常: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        else:
            api_logger.error(f"❌ Unknown mode: {mode}")
            return None
        
        if not tts_speeches:
            return None
        
        # 合并音频
        audio_data = torch.concat(tts_speeches, dim=1)
        
        # 处理速度变化
        if speed != 1.0:
            # 保存为临时文件
            sample_rate = getattr(cosyvoice, 'sample_rate', 22050)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
                torchaudio.save(tmp_input.name, audio_data, sample_rate, format='wav')
                temp_input_path = tmp_input.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                temp_output_path = tmp_output.name
            
            # 使用 FFmpeg 变速
            if speed_change_ffmpeg(temp_input_path, speed, temp_output_path):
                # 读取变速后的音频
                audio_data, _ = torchaudio.load(temp_output_path)
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            else:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
                api_logger.warning("⚠️ Speed change failed, returning original audio")
        
        # 保存到内存缓冲区
        buffer = io.BytesIO()
        sample_rate = getattr(cosyvoice, 'sample_rate', 22050)
        torchaudio.save(buffer, audio_data, sample_rate, format='wav')
        buffer.seek(0)
        
        # 计算音频时长（秒）
        audio_duration = audio_data.shape[1] / sample_rate if audio_data.numel() > 0 else 0
        audio_size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
        
        # 计算推理耗时和RTF
        end_time = time.time()
        total_time = end_time - start_time
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        
        api_logger.info(f"✅ 推理成功 | ⏱️ 耗时: {total_time:.2f}s | ⚡ RTF: {rtf:.4f} | 🎵 时长: {audio_duration:.2f}s | 💾 大小: {audio_size_mb:.2f}MB")
        
        return buffer
    
    except Exception as e:
        api_logger.error(f"❌ [推理] 总体异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 主程序入口 ====================

if __name__ == "__main__":
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
        help='设置最小文本长度，低于该长度的请求将被跳过 (默认: 0，不限制) 但是低了于4个字符可能效果不好甚至导致推理不出来'
    )
    
    args = parser.parse_args()
    
    # 初始化应用
    config_file = args.config
    if not os.path.isabs(config_file):
        config_file = os.path.join(ROOT_DIR, '..', config_file)
    
    # 确保指定的是一个 JSON 文件
    if not config_file.endswith('.json'):
        print(f"❌ Error: --config must point to a .json file, got: {config_file}")
        sys.exit(1)
    
    # 初始化角色配置（只读取指定的文件）
    character_config = CharacterConfig(config_file)
    
    # 设置最小文本长度
    set_min_text_length(args.min_text_length)
    
    # 启动前加载模型
    api_logger.info(f"📦 Loading CosyVoice model...")
    try:
        cosyvoice = load_cosyvoice_model()
        api_logger.info(f"✅ Model loaded successfully")
    except Exception as e:
        api_logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 启动服务器
    print(f"\n🚀 Starting CosyVoice3 API Server...")
    print(f"📍 Host: {args.host}:{args.port}")
    print(f"🔗 Health check: http://{args.host}:{args.port}/api/health")
    
    # 禁用 Flask 默认的 banner
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
