from fastapi import APIRouter
from pydantic import BaseModel, Field

from .common import api_module, audio_response, json_response

router = APIRouter(tags=["Tavern"])


class TavernSpeechRequest(BaseModel):
    text: str = Field(..., description="要生成的文本")
    speaker: str = Field(..., description="角色名称")
    speed: float = Field(1.0, description="语速倍数")
    streaming: int | bool | None = Field(None, description="SillyTavern兼容字段")


@router.get("/speakers", summary="List speakers (SillyTavern compatible)")
@router.get("/api/characters", include_in_schema=False)
async def get_speakers():
    api = api_module()
    try:
        return json_response(api.build_tavern_speakers())
    except Exception as exc:
        return json_response({"error": str(exc)}, status_code=500)


@router.post("/", summary="Generate speech (SillyTavern compatible)")
async def tavern_speech(payload: TavernSpeechRequest):
    api = api_module()
    try:
        if api.cosyvoice is None:
            return json_response({"error": "模型未加载"}, status_code=500)
        if api.character_config is None:
            return json_response({"error": "角色配置未加载"}, status_code=500)

        text = api.prepare_request_text(payload.text, field_name="text")
        char_config = api.character_config.get_character(payload.speaker)
        if not char_config:
            return json_response({"error": f"未找到角色: {payload.speaker}"}, status_code=404)

        api.api_logger.info(
            "📝 Tavern request: speaker=%s, speed=%s, text_len=%s",
            payload.speaker,
            payload.speed,
            len(text),
        )
        audio_buffer = api._inference(text=text, char_config=dict(char_config), mode=None, speed=float(payload.speed))
        if audio_buffer is None:
            return json_response({"error": "生成音频失败"}, status_code=500)

        return audio_response(audio_buffer, "audio/wav", "speech.wav")
    except ValueError as exc:
        return json_response({"error": str(exc)}, status_code=400)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return json_response({"error": f"请求异常: {str(exc)[:150]}"}, status_code=500)
