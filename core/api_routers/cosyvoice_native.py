from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from .common import api_module, audio_response, json_response, parse_mixed_request

router = APIRouter(tags=["CosyVoice Native"])


class NativeSpeechRequest(BaseModel):
    text: str = Field(..., description="待合成文本，原样传给后端 tokenizer")
    mode: str | None = Field(None, description="zero_shot | cross_lingual | instruct")
    profile: str | None = Field(None, description="可选，本地预设名称")
    character_name: str | None = Field(None, description="兼容字段，等同于 profile")
    prompt_audio_path: str | None = Field(None, description="服务端本地参考音频路径")
    prompt_text: str | None = Field(None, description="zero_shot 必填")
    instruct_text: str | None = Field(None, description="instruct 必填")
    prompt_lang: str | None = Field(None, description="预留字段")
    speed: float = Field(1.0, description="语速倍数")
    response_format: str = Field("wav", description="输出格式")


def _payload_dict(payload: NativeSpeechRequest) -> dict[str, Any]:
    return payload.model_dump(exclude_none=True)


async def _run_native_request(payload: dict[str, Any], uploaded_audio: UploadFile | None = None):
    api = api_module()
    temp_prompt_audio_path = None
    try:
        if api.cosyvoice is None:
            return json_response({"error": "模型未加载"}, status_code=500)

        if uploaded_audio is not None and uploaded_audio.filename:
            temp_prompt_audio_path = api.save_uploaded_prompt_audio(uploaded_audio.filename, await uploaded_audio.read())

        text = api.prepare_request_text(
            api.extract_request_field(payload, "text", "input", default=""),
            field_name="text",
        )
        speed = float(api.extract_request_field(payload, "speed", default=1.0) or 1.0)
        response_format = str(
            api.extract_request_field(payload, "response_format", "format", default="wav") or "wav"
        ).strip().lower()
        runtime_config, requested_profile = api.build_runtime_char_config(
            payload,
            uploaded_prompt_audio_path=temp_prompt_audio_path,
        )

        api.api_logger.info(
            "📝 Native request: profile=%s, mode=%s, format=%s, speed=%s, text_len=%s",
            requested_profile or runtime_config.get("name", ""),
            api.get_mode_label(runtime_config.get("mode")),
            response_format,
            speed,
            len(text),
        )
        audio_buffer = api._inference(
            text=text,
            char_config=runtime_config,
            mode=runtime_config.get("mode"),
            speed=speed,
        )
        if audio_buffer is None:
            return json_response({"error": "生成音频失败"}, status_code=500)

        converted_buffer, mime_type = api.convert_audio_buffer_format(audio_buffer, response_format)
        return audio_response(converted_buffer, mime_type, f"speech.{response_format}")
    except ValueError as exc:
        return json_response({"error": str(exc)}, status_code=400)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return json_response({"error": f"请求异常: {str(exc)[:150]}"}, status_code=500)
    finally:
        api.cleanup_temp_file(temp_prompt_audio_path)


@router.get("/cosyvoice/meta", summary="Describe native CosyVoice capabilities")
@router.get("/api/v1/tts/cosyvoice/meta", include_in_schema=False)
@router.get("/api/tts/cosyvoice/meta", include_in_schema=False)
async def cosyvoice_meta():
    api = api_module()
    try:
        return json_response(api.build_native_cosyvoice_meta())
    except Exception as exc:
        return json_response({"error": str(exc)}, status_code=500)


@router.get("/cosyvoice/profiles", summary="List local CosyVoice profiles")
async def cosyvoice_profiles():
    api = api_module()
    try:
        return json_response({"object": "list", "data": api.build_profile_items()})
    except Exception as exc:
        return json_response({"error": str(exc)}, status_code=500)


@router.post("/cosyvoice/speech", summary="Generate speech with native CosyVoice JSON API")
async def cosyvoice_speech(payload: NativeSpeechRequest):
    return await _run_native_request(_payload_dict(payload))


@router.post("/cosyvoice/speech/upload", summary="Generate speech with uploaded prompt audio")
async def cosyvoice_speech_upload(
    text: str = Form(...),
    mode: str | None = Form(None),
    profile: str | None = Form(None),
    character_name: str | None = Form(None),
    prompt_audio_path: str | None = Form(None),
    prompt_audio: UploadFile | None = File(None),
    prompt_text: str | None = Form(None),
    instruct_text: str | None = Form(None),
    prompt_lang: str | None = Form(None),
    speed: float = Form(1.0),
    response_format: str = Form("wav"),
):
    payload = {
        "text": text,
        "mode": mode,
        "profile": profile,
        "character_name": character_name,
        "prompt_audio_path": prompt_audio_path,
        "prompt_text": prompt_text,
        "instruct_text": instruct_text,
        "prompt_lang": prompt_lang,
        "speed": speed,
        "response_format": response_format,
    }
    return await _run_native_request(payload, uploaded_audio=prompt_audio)


@router.post("/api/v1/tts/cosyvoice", include_in_schema=False)
@router.post("/api/tts/cosyvoice", include_in_schema=False)
@router.post("/api/tts", include_in_schema=False)
async def legacy_native_speech(request: Request):
    payload, uploaded_audio = await parse_mixed_request(request)
    return await _run_native_request(payload, uploaded_audio=uploaded_audio)
