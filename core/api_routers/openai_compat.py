from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .common import api_module, audio_response, json_response, openai_error

router = APIRouter(tags=["OpenAI Compatible"])


class OpenAiSpeechRequest(BaseModel):
    model: str = Field("cosyvoice-openai-tts", description="模型名，占位兼容字段")
    input: str = Field(..., description="待合成文本")
    voice: str | dict[str, Any] = Field(..., description="角色名称或 {id, name} 对象")
    instructions: str | None = Field(None, description="可选，附加语气/风格指令")
    speed: float = Field(1.0, description="语速倍数")
    response_format: str = Field("mp3", description="输出格式")
    mode: str | None = Field(None, description="可选，覆盖原生推理模式")


@router.get("/v1/models", summary="List available models")
async def get_models():
    api = api_module()
    try:
        return json_response({"object": "list", "data": api.build_model_items()})
    except Exception as exc:
        return json_response({"error": str(exc)}, status_code=500)


@router.get("/v1/audio/voices", summary="List voices (OpenAI-style extension)")
@router.get("/v1/audio/speakers", include_in_schema=False)
async def get_voices():
    api = api_module()
    try:
        return json_response({"object": "list", "data": api.build_speaker_items()})
    except Exception as exc:
        return json_response({"error": str(exc)}, status_code=500)


@router.post("/v1/audio/speech", summary="Generate speech (OpenAI compatible)")
async def openai_audio_speech(payload: OpenAiSpeechRequest):
    api = api_module()
    try:
        if api.cosyvoice is None:
            return openai_error("模型未加载", status_code=500, error_type="server_error")
        if api.character_config is None:
            return openai_error("角色配置未加载", status_code=500, error_type="server_error")

        text = api.prepare_request_text(payload.input, field_name="input")
        voice_name = api.extract_voice_name(payload.voice)
        if not voice_name:
            return openai_error("voice 不能为空", status_code=400)

        char_config = api.character_config.get_character(voice_name)
        if not char_config:
            return openai_error(f"未找到角色: {voice_name}", status_code=404)

        char_config = dict(char_config)
        if payload.instructions:
            char_config["instruct_text"] = payload.instructions
            char_config["mode"] = "instruct"

        response_format = (payload.response_format or "mp3").strip().lower()
        override_mode = api.normalize_mode_name(payload.mode)

        api.api_logger.info(
            "📝 OpenAI request: voice=%s, format=%s, speed=%s, text_len=%s",
            voice_name,
            response_format,
            payload.speed,
            len(text),
        )
        audio_buffer = api._inference(
            text=text,
            char_config=char_config,
            mode=override_mode,
            speed=float(payload.speed),
        )
        if audio_buffer is None:
            return openai_error("生成音频失败", status_code=500, error_type="server_error")

        converted_buffer, mime_type = api.convert_audio_buffer_format(audio_buffer, response_format)
        return audio_response(converted_buffer, mime_type, f"speech.{response_format}")
    except ValueError as exc:
        return openai_error(str(exc), status_code=400)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return openai_error(f"请求异常: {str(exc)[:150]}", status_code=500, error_type="server_error")
